# **EDVERTO: The Chatbot**

EDVERTO is a chatbot powered by a custom-built Transformer-based model designed for intent classification and response generation. The chatbot is trained on an `intents.json` file and responds to user input based on predefined intents and patterns.

---

## **Features**
- **Transformer Model**: Leverages a custom Transformer-based architecture for intent recognition.
- **Preprocessing**: Includes text preprocessing and tokenization for efficient training and inference.
- **User-Friendly Interface**: Built using **Streamlit**, providing an interactive and modern UI.
- **Conversation Logging**: Logs conversations with timestamps for tracking history.
- **Flexible Responses**: Maps intents to predefined responses, ensuring relevant and contextually appropriate replies.

---

## **How It Works**
1. **Dataset**: 
   - The chatbot uses an `intents.json` file, structured with:
     - **Tags**: Representing intents.
     - **Patterns**: Training phrases for each intent.
     - **Responses**: Predefined responses mapped to each intent.
     
2. **Model**:
   - A Transformer-based model processes input patterns, classifies them into intents, and selects an appropriate response.
   - Architecture highlights:
     - Embedding layer for text representation.
     - Multi-head attention mechanism.
     - Residual connections and Layer Normalization.
     - Fully connected layers for classification.

3. **Chat Interface**:
   - Users interact with the chatbot via a Streamlit-based interface.
   - Sidebar navigation for:
     - **Home**: Start chatting.
     - **Conversation History**: View past conversations.
     - **About EDVERTO**: Learn more about the project.

---

## **Setup Instructions**
### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/edverto-chatbot.git
cd edverto-chatbot
```

### **2. Install Dependencies**
Make sure you have Python 3.8+ installed. Install the required packages using:
```bash
pip install -r requirements.txt
```

### **3. Prepare the Dataset**
- Place your `intents.json` file in the root directory.
- Ensure the file is structured as follows:
  ```json
  [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Hey there"],
      "responses": ["Hello!", "Hi there!", "Greetings!"]
    },
    ...
  ]
  ```

### **4. Run the Application**
Launch the chatbot using Streamlit:
```bash
streamlit run Edverto.py
```

---

## **Folder Structure**
```
.
├── Edverto.py           # Main application file
├── intents.json         # Intents dataset
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## **Usage**
1. Start the application by running the Streamlit command.
2. Navigate to the **Home** tab to begin chatting.
3. View conversation logs in the **Conversation History** tab.
4. Learn more about the chatbot in the **About EDVERTO** section.

---

## **Customization**
- **Dataset**: Modify the `intents.json` file to include your own intents, patterns, and responses.
- **Model**: Adjust model parameters in the `create_transformer_model` function to suit your dataset or add advanced features.
- **UI**: Customize the Streamlit interface to enhance user experience.

---

## **Demo**
![EDVERTO Chatbot Demo](https://github.com/Abhijat-M/chatbot_edverto/blob/main/demo.png)

---

## **Future Enhancements**
- Add multi-turn conversational capabilities.
- Integrate a database for storing conversation logs.
- Deploy the application using Docker or cloud services.

---

## **Contributing**
Contributions are welcome! Feel free to submit issues or pull requests.

---

## **License**
This project is licensed under the MIT License. See `LICENSE` for more details.

---

**Developed with ❤️ using Python, TensorFlow, and Streamlit**.
