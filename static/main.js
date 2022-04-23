const msgInputForm = document.getElementById('msg-input-form');
const msgInputField = document.getElementById('msg-input-field');
const chatArea = document.getElementById('chat-area');

// Icons made by Freepik from www.flaticon.com
// const BOT_IMG = 'https://image.flaticon.com/icons/svg/327/327779.svg';
// const PERSON_IMG = 'https://image.flaticon.com/icons/svg/145/145867.svg';
const BOT_NAME = 'Chatbot Ben';
const PERSON_NAME = 'You';

msgInputForm.addEventListener('submit', function (event) {
    event.preventDefault();
    const pattern = /[\w]/

    var msgText = msgInputField.value;

    if (pattern.test(msgText.charAt(msgText.length - 1))) {
        msgText = msgText + ".";
    }

    msgText = msgText.charAt(0).toUpperCase() + msgText.slice(1);

    if (!msgText) return;
    appendMessage(PERSON_NAME, 'receiver', msgText);
    msgInputField.value = '';
    botResponse(msgText);
});

function appendMessage(name, side, text) {
    //   Simple solution for small apps
    const msgHTML = `
<div class="msg ${side}-msg">
  <h3>${name}</h3>
  <p>${text}</p>
</div>
`;

    let msg = document.createElement('div');
    msg.innerHTML = msgHTML;
    chatArea.appendChild(msg);
    window.scrollBy(0, msg.clientHeight)
}

function botResponse(rawText) {
    // Bot Response
     $.get('/get', { msg: rawText }).done(function (data) {
         appendMessage(BOT_NAME, 'sender', data);
     });
}

// Utils
function get(selector, root = document) {
    return root.querySelector(selector);
}

function formatDate(date) {
    const h = '0' + date.getHours();
    const m = '0' + date.getMinutes();

    return `${h.slice(-2)}:${m.slice(-2)}`;
}
