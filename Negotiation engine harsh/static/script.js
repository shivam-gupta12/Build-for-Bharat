document.getElementById('checkbox').addEventListener('change', function(event){
    if(this.checked) {
        document.body.classList.add("dark-mode");
    } else {
        document.body.classList.remove("dark-mode");
    }
});

function createBubble(text) {
    var bubble = document.createElement('div');
    bubble.classList.add('chat-bubble');
    bubble.textContent = text;
    var chatContainer = document.getElementById('chatbox');
    chatContainer.appendChild(bubble); // Append bubble at the end
  
    // Scroll to the bottom of the chat container
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }
  

document.querySelector('.send-button').addEventListener('click', function() {
    var input = document.getElementById('user_input');
    if (input.value.trim() !== '') { // Check if the input is not just whitespace
      createBubble(input.value);
      input.value = ''; // Clear the input field
    }
});
  
  

