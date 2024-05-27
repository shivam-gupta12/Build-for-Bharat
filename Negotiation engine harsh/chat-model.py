from flask import Flask, render_template, request, session, redirect, url_for, copy_current_request_context, send_from_directory, abort
from flask_socketio import join_room, leave_room, send, SocketIO
import random
import torch
import json
from flask_socketio import emit
import markdown
from transformers import BertTokenizer, BertForSequenceClassification
import google.generativeai as genai 
import string
from threading import Thread
import os
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import uuid
from flask import jsonify
import re
from report_generate import generate_pdf_report


print('app starting')

app = Flask(__name__)
app.config["SECRET_KEY"] = "hjhjsdahhds"
app.config["UPLOAD_FOLDER"] = "/Users/damodargupta/Desktop/NEGOTIATION ENGINE/uploads"
socketio = SocketIO(app, cors_allowed_origins="*", engineio_logger=True, logger=True, async_mode='eventlet')


if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])
    
rooms = {}

last_proposed_compromise = ""
def generate_unique_code(length):
    while True:
        characters = string.ascii_letters + string.digits
        code = "".join(random.choice(characters) for _ in range(length))
        
        if code not in rooms:  
            break
    
    return code

@app.route("/", methods=["POST", "GET"])
def home():
    session.clear()
    if request.method == "POST":
        name = request.form.get("name")
        code = request.form.get("code")
        create = request.form.get("create", False)

        if not name:
            return {"error": "Please enter a name."}, 400

        if create and not code:
            room, new_var = new_func()
            new_var
            rooms[room] = {"members": 0, "messages": []}
            session["room"] = room
            session["name"] = name
            return {"redirect": url_for("room")}

        elif code in rooms:
            session["room"] = code
            session["name"] = name
            return {"redirect": url_for("room")}
        else:
            return {"error": "Room does not exist."}, 404

    return render_template("home.html")

def new_func():
    new_var = room = generate_unique_code(10)
    return room,new_var

@app.route('/fetch-report/<filename>')
def fetch_report(filename):
    # Ensure that the directory does not traverse to subdirectories
    if ".." in filename or filename.startswith("/"):
        return "Invalid file path", 400
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/room")
def room():
    room = session.get("room")
    if room is None or session.get("name") is None or room not in rooms:
        return redirect(url_for("home"))

    return render_template("room.html", code=room, messages=rooms[room]["messages"])

room_data = {}

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'pdf' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['pdf']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            reader = PdfReader(filepath)
            text = ''.join(page.extract_text() + ' ' for page in reader.pages if page.extract_text())

            # Save the extracted text associated with the room
            room = session.get("room")
            room_data[room] = text.strip()  # Save extracted text to global dictionary
        except Exception as e:
            return jsonify({"error": "Failed to read PDF: " + str(e)}), 500

        return jsonify({"success": "File successfully uploaded", "filepath": url_for('uploaded_file', filename=filename)}), 200
    else:
        return jsonify({"error": "Invalid file type. Only PDF files are allowed."}), 400


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

certainty_model_path = 'Negotiation engine harsh/models/certainty_prediction_model'
certainty_model = BertForSequenceClassification.from_pretrained(certainty_model_path)
certainty_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
certainty_model.to(device)

# phi2_model_name = "models/phi-2"
# phi2_model = AutoModelForCausalLM.from_pretrained(phi2_model_name)
# phi2_tokenizer = AutoTokenizer.from_pretrained(phi2_model_name, trust_remote_code =True)
# phi2_pipeline = pipeline("text-generation", model=phi2_model, tokenizer=phi2_tokenizer)
# phi2_model.to(device)

# SYSTEM_PROMPT = """
# Analyze a negotiation dialogue to identify non-negotiable terms and suggest alternative strategies.
# """

# generation_config = GenerationConfig.from_pretrained(phi2_model_name)
# generation_config.max_new_tokens = 512
# generation_config.temperature = 0.001
# generation_config.do_sample = True

# streamer = TextStreamer(phi2_tokenizer, skip_prompt=True, skip_special_tokens=True)

# llm = pipeline(
#     "text-generation",
#     model=phi2_model,
#     tokenizer=phi2_tokenizer,
#     # max_new_tokens = 512,
#     eos_token_id=phi2_tokenizer.eos_token_id,
#     pad_token_id=phi2_tokenizer.eos_token_id,
#     streamer=streamer,
#     )

# def create_suggestion_prompt(prompt: str, system_prompt: str = SYSTEM_PROMPT)-> str:
#     """
#     Creates a prompt for the phi-2 model to generate alternative negotiation strategies based on the conversation.

#     Parameters:
#     conversation (list of str): The conversation history, where each item is a message string.

#     Returns:
#     str: A prompt string for the phi-2 model.
#     """
#     if not system_prompt:
#         return cleandoc(
#             f"""
#         Instruct: {prompt}
#         Output:
#         """
#         )
#     return cleandoc(
#         f"""
#         Instruct: {system_prompt} {prompt}
#         Output:
#         """
#     )


CERTAINTY_SEQUENCE_THRESHOLD = 1
last_active_user = {}


def background_task(func, *args, **kwargs):
    """ Run a background task in a separate thread. """
    thread = Thread(target=func, args=args, kwargs=kwargs)
    thread.start()
    return thread



@socketio.on("message")
def handle_message(data):
    print("calling handle_message function")
    room = session.get("room")
    if room not in rooms:
        return
    

    user = session.get("name")
    message_text = data["data"]
    
    last_active_user[room] = user
    
    if "messages_by_user" not in rooms[room]:
        rooms[room]["messages_by_user"] = {}
    if "certainty_history_by_user" not in rooms[room]:
        rooms[room]["certainty_history_by_user"] = {}

    if user not in rooms[room]["messages_by_user"]:
        rooms[room]["messages_by_user"][user] = []
    if user not in rooms[room]["certainty_history_by_user"]:
        rooms[room]["certainty_history_by_user"][user] = []

    rooms[room]["messages_by_user"][user].append(message_text)
    
    all_messages = sum(rooms[room]["messages_by_user"].values(), [])
    print('all messages :' , all_messages)
    
    # @copy_current_request_context
    # def background_generate_middle_ground(all_messages, room):
    #     generate_middle_ground(all_messages, room)

    # # Create a copy of the current request context to use in the background task
    # @copy_current_request_context
    # def background_generate_and_send_suggestion(user_messages, room, user):
    #     generate_and_send_suggestion(user_messages, room, user)
        
    # background_task(background_generate_middle_ground, all_messages, room)
    
    generate_middle_ground(all_messages, room)
    
    user_messages = rooms[room]["messages_by_user"][user]
    overall_certainty, individual_predictions = predict_certainty_conversation(user_messages, certainty_model, certainty_tokenizer)
    rooms[room]["certainty_history_by_user"][user].append(overall_certainty)
    print('overall certainity ', overall_certainty)
    for user, certainty_history in rooms[room]["certainty_history_by_user"].items():
        # if len(certainty_history) >= CERTAINTY_SEQUENCE_THRESHOLD and all(certainty_history[-CERTAINTY_SEQUENCE_THRESHOLD:]):
            # if len(certainty_history) == CERTAINTY_SEQUENCE_THRESHOLD or not all(certainty_history[-CERTAINTY_SEQUENCE_THRESHOLD-1:-1]):
        if len(certainty_history) >= CERTAINTY_SEQUENCE_THRESHOLD:
            generate_and_send_suggestion(user_messages, room, user)

    send({"name": session.get("name"), "message": message_text, "certainty": "Certain" if overall_certainty else "Not Certain"}, to=room)

    
def generate_and_send_suggestion(user_messages, room, user):
    print('calling function generate prompt')
    conversation_context = " ".join(user_messages[:-1]) if len(user_messages) > 1 else "The conversation has focused primarily on price negotiations."
    non_negotiable_statement = user_messages[-1]
    
    genai.configure(api_key="AIzaSyCzRrKO-krrIN0YfSgmj_MQJ9nqnXLFLdI")
    model = genai.GenerativeModel('gemini-pro')

    prompt = f"""
    Given the negotiation context below, identify the key negotiable term from the most recent statement and suggest an alternative strategy other than the negotiable term. The strategy should consider other aspects or terms that could be negotiable, aiming for a win-win outcome. It's important to communicate the suggestions in a way that indicates possibilities rather than certainties. Keep the responses short.

    Previous negotiation context:
    {conversation_context}

    Most recent, non-negotiable statement:
    {non_negotiable_statement}

    Task:
    firstly analyse the statements and decide whether an intervention is necessary at this point or not, if yes then :
        1. Identify the term which can be negotiated upon from the most recent statement.
        2. If no negotiable term is there, just respond with "No comment". Otherwise, Propose an alternative negotiation strategy other than the negotiable term while addressing the negotiation's key aspects.
        3. keep the responses short such that they are easy and fast to read for the user.
    """

    
    config = {
        "max_output_tokens": 216,
        "temperature" : 0.3
    }
    responses = model.generate_content(prompt, generation_config=config)
    
    text_responses = []

    # Iterate through each candidate in the responses
    for candidate in responses.candidates:
        # Concatenate the text of all parts for this candidate
        candidate_text = ''.join(part.text for part in candidate.content.parts)
        
        # Format the text using HTML for better structure
        formatted_response = format_response_as_html(candidate_text)
        text_responses.append(formatted_response)
    
    final_response_html = '<hr>'.join(text_responses)
    
    print('Suggestions generated by model (HTML formatted):', final_response_html)
    if "No comment" not in responses.text:
        formatted_response = to_html(responses.text)
        # Find the target user as the user who is not the last active user
        target_user = next((user for user in user_sessions[room] if user != last_active_user.get(room)), None)
        
        if target_user:
            target_session_id = user_sessions[room].get(target_user)
            if target_session_id:
                # Send the suggestion to the target user
                send({"name": "System", "message": formatted_response, "messageType": "suggestion"}, room=target_session_id)
                print(f"Suggestion sent to {target_user}.")
            else:
                print(f"Session ID for {target_user} not found.")
        else:
            print("No suitable target user found in the room.")

def to_html(markdown_text):
    html_content = markdown.markdown(markdown_text)
    left_aligned_html = f'{html_content}'
    return left_aligned_html


def generate_middle_ground(user_messages, room):
    global last_proposed_compromise
    print('middle ground function has been called')
    genai.configure(api_key="AIzaSyCHhvLttGJJynPDImQ3NGkxb4d7PTD4hKI")
    model = genai.GenerativeModel('gemini-pro')
    
    # Combine user messages into a single string for analysis
    combined_terms = '\n\n'.join(user_messages)

    pdf_text = room_data.get(room, '')

    prompt = f"""
        Given a negotiation between a seller and a buyer, we aim to mediate and find a mutually beneficial compromise based on the terms provided by each party. Here, the initial terms from a foundational PDF document and the ongoing negotiation terms need to be considered:

        Initial Terms from PDF (use as base):
        {pdf_text}

        New Terms Proposed During Negotiation:
        {combined_terms}

        Instructions:
        1. Analyze the initial terms and the new terms proposed during negotiation.
        2. Identify key differences and potential areas for compromise based on both sets of terms.
        3. Propose a compromise that strictly incorporates elements from both the initial and newly proposed terms, considering constraints such as price and additional features.
        4. The output should be a concise, unbiased compromise, derived strictly from the provided terms.

        If sufficient information is available to propose a meaningful compromise, output the compromise. If not, respond with "No comment".
        """

    # Assuming genai and model are configured as shown earlier
    config = {
        "max_output_tokens": 216,
        "temperature": 0.3,
    }
    response = model.generate_content(prompt)

    if "No comment" not in response.text:
        formatted_response = to_html(response.text)
        last_proposed_compromise = formatted_response
        send({"name": "System", "message": formatted_response, "messageType": "middle-ground"}, to=room, html=True)
    else:
        print('not enough information')
        


def find_json_string(raw_string):
    # This regex pattern attempts to find the most extended string that starts with '{' and ends with '}'.
    # It's somewhat simplified and assumes that the JSON does not contain any strings with unescaped '}' characters.
    pattern = r'\{.*\}'
    match = re.search(pattern, raw_string, re.DOTALL)
    if match:
        return match.group(0)  # Return the matched JSON string
    return None  # If no match is found, return None

def summarize(user_messages, room):
    conversation_context = " ".join(user_messages[:-1]) if len(user_messages) > 1 else "The conversation has focused primarily on price negotiations."
    non_negotiable_statement = user_messages[-1]
    
    genai.configure(api_key="AIzaSyCzRrKO-krrIN0YfSgmj_MQJ9nqnXLFLdI")
    model = genai.GenerativeModel('gemini-pro')

    prompt = f"""
    Analyse the messages from both buyer and seller parties and determine the status of the negotiation which can either of the three i.e accepted, rejected or no decision made. Also summarize the conversation in a brief way to determine actual conclusion.

    Previous negotiation context:
    {conversation_context}

    Most recent, non-negotiable statement:
    {non_negotiable_statement}

    Task:
        1. Determine the buyer and seller messages from all messages and analyse them to gain understanding of the negotiation.
        2. After you have analysed all the messages from both sides, determine the status of the negotiation which can either of the three i.e accepted, rejected or no decision made. 
        3. Now summarize the entire conversation in a brief way to find out the actual conclusion of the negotiation between both parties.
        4. Give your output only in the given JSON format :
            {{
                "status" : "status of the conversation (accepted, rejected or no decision made)",
                "summary" : "conclusion of the conversation"
            }}
    """

    generation_config = {
        "max_output_tokens": 216,
        "temperature": 0.3
    }
    
    try:
        response = model.generate_content(prompt, generation_config=generation_config)
        summary_whole = response.text
        summary_whole = find_json_string(summary_whole)
        print(summary_whole, type(summary_whole))
        summary_whole = json.loads(summary_whole)
        print(type(summary_whole))
        print(summary_whole)
        emit('summary_result', {"name": "System", "message": {"status": summary_whole["status"], "summary": summary_whole["summary"]}, "messageType": "status and summary"}, room=room)
        return summary_whole
    except Exception as e:
        emit('summary_error', {"error": str(e)}, room=room)
    
def format_response_as_html(text):
    """
    Converts a structured text response into HTML with left-aligned text for better readability.
    """
    html_response = '<div style="text-align: left;">'
    for line in text.split('\n'):
        if line.strip():  
            html_response += f'<p>{line}</p>'
    html_response += '</div>'
    return html_response
    
def predict_certainty_conversation(conversation, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    cumulative_predictions = []
    weights = [0.5 ** i for i in range(len(conversation))]
    weights.reverse()

    for dialogue in conversation:
        inputs = tokenizer(dialogue, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=1)
        cumulative_predictions.append(predicted_class.item())

    if len(cumulative_predictions) > 1:
        weighted_average = sum([a * b for a, b in zip(cumulative_predictions[:-1], weights[:-1])]) / sum(weights[:-1])
    else:
        weighted_average = 0  

    if cumulative_predictions[-1] == 1:  
        overall_certainty = 1
    else:
        overall_certainty = 1 if weighted_average > 0.5 else 0

    return overall_certainty, cumulative_predictions

    
user_sessions = {}  

@socketio.on('connect')
def on_connect():
    session_id = request.sid  
    room = session.get("room")
    name = session.get("name")
    if room and name:
        join_room(room)
        if room not in user_sessions:
            user_sessions[room] = {}
        user_sessions[room][name] = session_id
        print('users in a session ' , user_sessions)
        send({"name": name, "message": "has entered the room"}, room=room)

@socketio.on("disconnect")
def disconnect():
    room = session.get("room")
    name = session.get("name")
    leave_room(room)
    if room in rooms:
        rooms[room]["members"] -= 1
        if rooms[room]["members"] <= 0:
            del rooms[room]
    
    send({"name": name, "message": "has left the room"}, to=room)
    print(f"{name} has left the room {room}")
    
@socketio.on("end_conversation")
def end_conversation():
    room = session.get("room")
    if room in rooms:
        try:
            all_messages = sum(rooms[room]["messages_by_user"].values(), [])
            summary_data = summarize(all_messages, room)  # Your summarize function
            print("summary ", summary_data)
            print("last proposed compromise ", last_proposed_compromise)
            report_data = {
                **summary_data,
                "last_proposed_compromise": last_proposed_compromise
            }
            report_path = f"{app.config['UPLOAD_FOLDER']}/Negotiation_Summary_Report_{uuid.uuid4().hex}.pdf"
            generate_pdf_report(report_data, report_path)
            emit('report_generated', {'report_url': url_for('fetch_report', filename=f"{os.path.basename(report_path)}")}, room=room)
        except Exception as e:
            print("Error during report generation:", str(e))
            emit('summary_error', {"error": str(e)}, room=room)

if __name__ == "__main__":
    socketio.run(app,port=8080)