import os
from slack_bolt import App
from slack_sdk.web import WebClient
import ssl as ssl_lib
import certifi
import re
from  llindex import LL_INDEX

openai_api_key=os.environ.get("OPENAI_API_KEY")

app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")  
)

channel_messages = {}

def _remove_mention(text):
    return re.compile(r'<@[^>]+>').sub('', text).lstrip()

def _get_request_channel(text):
    if re.search(r'<#[A-Z0-9]+\|(.*?)>', text):
        return re.search(r'<#[A-Z0-9]+\|(.*?)>', text).group(1)
    else:
        return None

def _remove_request_channel(text):
    return re.compile(r'<#[A-Z0-9]+\|(.*?)>').sub('', text).lstrip()

@app.event("message")
def handle_message_events(body, logger):
  pass

@app.event("app_mention")
def handle_mention(body, say, logger):
    past_messages = []
    result = {}

    event_user = body.get("event", {}).get("user")
    event_ts = body.get("event", {}).get("ts")
    event_text = _remove_mention(body.get("event", {}).get("text"))
    event_channel = body.get("event", {}).get("channel")
    supported_qas = ['engineering','facilities','hr','it','general']

    if event_channel not in channel_messages:
      channel_messages[event_channel] = LL_INDEX()
    ll_index = channel_messages[event_channel]

    channel_name = _get_request_channel(event_text)
    
    print(channel_name)
    print(event_text)
    print(event_channel)
    print(event_user)

    if channel_name not in supported_qas:
        say(f"Please enter your query by mentioning the slack bot along with the any one of the following channels {supported_qas} followed by your query.")
        return  

    if event_text == "reset":
        say(f"Conversation reset.")
        return

    prompt = _remove_request_channel(event_text)
    print(prompt)

    say(f"<@{event_user}> Processing your request...")

    if channel_name == 'engineering':
        output = ll_index.fetch_openai_output(prompt)
    
    elif channel_name == 'facilities':
        output = "This is the response for FMCG query."
    elif channel_name == 'hr':
        output = ll_index.fetch_hr_qa_output(prompt)
    elif channel_name == 'it':
        output = "This is the response for IT query." 
    elif channel_name == 'general':
        output = "This is the response for General query."
  
    say(f"Here you go.")

    say(f"{output}")
    

if __name__ == "__main__":
    app.start(3000)
