from flask import Flask, render_template, request, session, redirect, url_for
from flask_socketio import join_room, leave_room, send, SocketIO
import random
from string import ascii_uppercase
import torch

from transformers import BertTokenizer, BertForSequenceClassification


app = Flask(__name__)
app.config["SECRET_KEY"] = "hjhjsdahhds"
socketio = SocketIO(app)

rooms = {}

def generate_unique_code(length):
    while True:
        code = ""
        for _ in range(length):
            code += random.choice(ascii_uppercase)
        
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
            room = generate_unique_code(4)
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


@app.route("/room")
def room():
    room = session.get("room")
    if room is None or session.get("name") is None or room not in rooms:
        return redirect(url_for("home"))

    return render_template("room.html", code=room, messages=rooms[room]["messages"])


model_path = 'Negotiation engine harsh/models/certainty_prediction_model'  
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


@socketio.on("message")
def handle_message(data):
    room = session.get("room")
    if room not in rooms:
        return

    # Extract the message text
    message_text = data["data"]

    # Analyze the message for certainty
    inputs = tokenizer(message_text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=1)
    certainty = "Certain" if predicted_class.item() == 1 else "Not Certain"

    content = {
        "name": session.get("name"),
        "message": message_text,
        "certainty": certainty
    }
    send(content, to=room)
    rooms[room]["messages"].append(content)
    print(f"{session.get('name')} said: {message_text} (Certainty: {certainty})")


@socketio.on("connect")
def connect(auth):
    room = session.get("room")
    name = session.get("name")
    if not room or not name:
        return
    if room not in rooms:
        leave_room(room)
        return
    
    join_room(room)
    send({"name": name, "message": "has entered the room"}, to=room)
    rooms[room]["members"] += 1
    print(f"{name} joined room {room}")

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

if __name__ == "__main__":
    socketio.run(app, debug=True , port=1221)