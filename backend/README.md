# Restaurant Booking Agent with Ollama

A conversational restaurant booking agent powered by Ollama (local LLM) that can handle natural language requests for restaurant reservations.

## Features

- 🤖 **Local LLM**: Uses Ollama for privacy and offline operation
- 🗣️ **Natural Language**: Understands conversational requests
- 📅 **Smart Date/Time Parsing**: Handles "tomorrow", "saturday", "7pm", etc.
- 🔄 **Context Awareness**: Remembers conversation state
- 🛠️ **Tool-based**: Uses LangChain tools for reliable operations
- 🌐 **REST API**: FastAPI backend for easy integration

## Prerequisites

1. **Ollama**: Install from [ollama.ai](https://ollama.ai)
2. **Python 3.8+**: For running the backend
3. **Restaurant API**: Mock API server for testing

## Quick Start

### 1. Install Ollama

```bash
# Download and install from https://ollama.ai
# Then pull the model
ollama pull llama3.2:3b
```

### 2. Setup the Backend

```bash
cd restaurant-booking-agent/backend

# Run the setup script
python setup.py

# Or manually install dependencies
pip install -r requirements.txt
```

### 3. Start the Server

```bash
python app.py
```

The server will start on `http://localhost:8000`

## API Usage

### Chat Endpoint

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I want to book a table for tomorrow at 7pm for 4 people",
    "session_id": "user123"
  }'
```

### Reset Session

```bash
curl -X POST http://localhost:8000/reset/user123
```

## Example Conversations

### Booking a Table

```
User: "I want to book a table"
Agent: "🎯 I'd be happy to help you make a reservation! 

To get started, I'll need a few details:
• Your name
• Date you'd like to visit (e.g., "tomorrow", "saturday", "2025-08-15")
• Time you prefer (e.g., "7pm", "19:30")
• Number of people in your party

What's your name?"

User: "My name is John Smith"
Agent: "What date would you like to visit? (e.g., 'tomorrow', 'saturday', 'next friday')"

User: "tomorrow"
Agent: "What time would you prefer? (e.g., '7pm', '19:30')"

User: "7pm"
Agent: "How many people will be in your party?"

User: "4 people"
Agent: "🎉 Perfect! Your reservation is confirmed!

📋 **Booking Reference:** ABC123
👤 **Name:** John Smith
📅 **Date:** 2025-01-07
🕐 **Time:** 19:00
👥 **Party:** 4 people

💡 **Important:** Please save your booking reference (ABC123) for future reference.

See you soon at TheHungryUnicorn! 🍽️"
```

### Checking Availability

```
User: "Check availability for this weekend"
Agent: "✅ Available time slots for 2025-01-11:
  • 12:00
  • 12:30
  • 13:00
  • 13:30
  • 19:00
  • 19:30

🎯 To book one of these times, just reply with your preferred time!"
```

## Configuration

Create a `.env` file in the backend directory:

```env
# Ollama Configuration
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434

# API Configuration
BOOKING_API_URL=http://localhost:8547
BOOKING_API_TOKEN=your_token_here

# Server Configuration
HOST=0.0.0.0
PORT=8000
```

## Architecture

### Core Components

1. **`app.py`**: FastAPI server with chat endpoints
2. **`agent_ollama.py`**: LangChain agent using Ollama LLM
3. **`tools.py`**: LangChain tools for booking operations
4. **`booking_client.py`**: API client for restaurant booking system

### Flow

1. User sends message to `/chat` endpoint
2. Message is processed by Ollama agent
3. Agent uses tools to perform actions (check availability, create booking, etc.)
4. Response is returned to user

## Tools Available

- **`check_availability`**: Check available times for a date
- **`create_booking`**: Create a new reservation
- **`get_booking`**: Retrieve booking details
- **`update_booking`**: Modify existing booking
- **`cancel_booking`**: Cancel a reservation

## Development

### Running Tests

```bash
# Test the setup
python setup.py

# Test individual components
python -c "from agent_ollama import BookingAgent; print('Agent works!')"
```

### Adding New Tools

1. Create a new tool class in `tools.py`
2. Add it to the tools list in `agent_ollama.py`
3. Update the prompt template if needed

### Customizing the Model

You can use different Ollama models by changing the `OLLAMA_MODEL` environment variable:

```bash
export OLLAMA_MODEL=llama3.2:8b  # Larger model
export OLLAMA_MODEL=llama3.2:1b  # Smaller model
export OLLAMA_MODEL=llama3.2:70b # Very large model
```

## Troubleshooting

### Ollama Not Running

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if needed
ollama serve
```

### Model Not Available

```bash
# List available models
ollama list

# Pull the required model
ollama pull llama3.2:3b
```

### API Connection Issues

Make sure the restaurant API server is running on the configured URL and port.

## Performance

- **Response Time**: Typically 1-3 seconds depending on model size
- **Memory Usage**: Varies by model (3B model ~2GB, 8B model ~4GB)
- **Concurrent Users**: Limited by Ollama server capacity

## Security

- All processing happens locally via Ollama
- No data sent to external LLM services
- API tokens should be kept secure
- Consider rate limiting for production use

## License

This project is for demonstration purposes. Please ensure compliance with Ollama and LangChain licenses for production use.
