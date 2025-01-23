const vscode = require("vscode");
const axios = require("axios");

function activate(context) {
  let disposable = vscode.commands.registerCommand(
    "codebase-chatbot.startChat",
    function () {
      const panel = vscode.window.createWebviewPanel(
        "codebaseChatbot",
        "Codebase Chatbot",
        vscode.ViewColumn.One,
        { enableScripts: true }
      );

      panel.webview.html = getWebviewContent();

      // Handle messages from the webview
      panel.webview.onDidReceiveMessage(
        async (message) => {
          if (message.command === "query") {
            try {
              const response = await axios.post("http://127.0.0.1:5000/query", {
                question: message.text,
              });
              panel.webview.postMessage({
                command: "response",
                text: response.data.response,
              });
            } catch (error) {
              panel.webview.postMessage({
                command: "response",
                text: "Error querying the backend.",
              });
            }
          }
        },
        undefined,
        context.subscriptions
      );
    }
  );

  context.subscriptions.push(disposable);
}

function deactivate() {}

module.exports = {
  activate,
  deactivate,
};
