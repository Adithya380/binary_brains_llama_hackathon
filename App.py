import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from datetime import date, datetime
from langchain_core.tools import tool
from langgraph.prebuilt import tools_condition
from typing import Optional
from langchain_core.messages import ToolMessage
from langchain_core.messages import ToolMessage
from typing import Literal
from IPython.display import Image, display
from langchain_core.messages import BaseMessage
from Test import change_prompt
from vision import predict_crop_disease
from openai import OpenAI
# import RAG
from typing import List
import getpass
import geocoder
import os
import base64
import json
import requests
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import soundfile as sf
import numpy as np
import io
os.environ['TAVILY_API_KEY']="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"

from langchain_nvidia_ai_endpoints import ChatNVIDIA

import httpx
llm = ChatOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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",
    model="meta/llama-3.1-70b-instruct",
    http_client=httpx.Client(verify=False)
)
client = llm

memory = MemorySaver()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
    return message


@tool()
def get_marketing_ideas():
        """ Use this tool to generate marketing ideas and marketing strategies for crop or animal produce.
        You can assume this tool as another agent or LLM that will generate the recommendations.
            Returns:
            str: recommendations for the given information."""

        tavily_tool = TavilySearchResults(max_results=4) #increased number of results

        #system message for generating the requirements
        # with open(r"prompts/sys_crop_rec.txt", "r", encoding="utf-8") as file:
        SYS_MAR_REC = """You are a helpful assistant to a farmer.
        
        You are an expert in marketing ideas and strategies for crops
        
         Follow the sequence of steps as given below to arrive at a logical answer:

         1. Get the current location using the get_current_location function
         
         2. Generate a few common marketing ideas for the farmer's crop
         
         3. Search the web using tavily_tool to find out if there are any crop or farm produce exhibitions near the location as idenfied above
         
         4. Finally give the marketing ideas and exhibition details from the above steps """

        react_agent_crop=create_react_agent(client,tools=[tavily_tool,get_current_location],state_modifier=SYS_MAR_REC)
        inputs={
            "messages":[
                {"role": "user",
                "content": f"Generate recommendations for the given information"}
            ]
            }
        message=print_stream(react_agent_crop.stream(inputs, stream_mode="values"))
        text=message.content

        return text

@tool()
def get_crop_recommendations():
        """ Use this tool to generate crop recommendations. Ask for the budget and land area mandatorily if the user does
           not mention it explicitly.
        You can assume this tool as another agent or LLM that will generate the recommendations.
        Returns:
            str: recommendations for the given information."""

        tavily_tool = TavilySearchResults(max_results=4) #increased number of results

        #system message for generating the requirements
        # with open(r"prompts/sys_crop_rec.txt", "r", encoding="utf-8") as file:
        SYS_CROP_REC = f"""You are a helpful assistant to farmers and farm owners.
                        You are an expert in recommending crops that are profitable and sustainable.
                        
                        Your job is to suggest crop recommendations to the farmer/farm owner by following the 
                        below sequential steps:

                        1. Get the location of the farmer using the get_current_location function

                        2. Get the current weather details of the location  using the get_weather function
                    
                        3.Give the crop recommendations using details from steps 1 and 2
                        
                        """

        react_agent_crop=create_react_agent(client,tools=[tavily_tool,get_weather,get_current_location],state_modifier=SYS_CROP_REC)
        inputs={
            "messages":[
                {"role": "user",
                "content": f"Generate recommendations for the given information"}
            ]
            }
        message=print_stream(react_agent_crop.stream(inputs, stream_mode="values"))
        text=message.content

        return text

@tool()
def get_farm_animal_recommendations():
        """ Use this tool to generate farm animal recommendations.
        You can assume this tool as another agent or LLM that will generate the recommendations.
        Returns:
            str: recommendations for the given information."""
        tavily_tool = TavilySearchResults(max_results=4) #increased number of results

        #system message for generating the requirements
        SYS_CROP_REC =f"""You are a helpful assistant to a farmer.
                        Give the farmer helpful, profitable and ecologically safe recommendations for farming
                        Use the tools provided to tailor make responses.
                        you have tools to access weather data and web results
                        Always keep in mind the budget of the farmer and the total area of cultivable land available.
                        your recommendation should contain
                        1. top 3 farm animals or birds or aquatic life that can be reared along with the crop. (ordered by benefit to crop)
                        2. expand all 3 Farm animal care emphasize on cost
                        3. common animal diseases which is prevalent in the area and mitigations
                      """

        react_agent_animal=create_react_agent(client,tools=[tavily_tool,get_weather],state_modifier=SYS_CROP_REC)
        inputs={
            "messages":[
                {"role": "user",
                "content": f"Generate recommendations for the given information"}
            ]
            }
        message=print_stream(react_agent_animal.stream(inputs, stream_mode="values"))
        text=message.content

        return text

@tool()
def get_farm_state():
        """ Use this tool to get the farm state details in the particular area of the farm.
        You can assume this tool as another agent or LLM that will generate the response.
        Returns:
            str: response for the given information."""
        # tavily_tool = TavilySearchResults(max_results=4) #increased number of results

        #system message for generating the requirements

        SYS_CROP_REC = f"""You are a helpful assistant to a farmer.
                        You are an expert in understanding the changes of landscape around a farm.

                        Please follow the following steps sequentially and give a final answer:

                        1. Get the current location of the farm using the get_current_location function

                        2. Understand the landscape changes that have happened around the farm as mentioned in {change_prompt}
                        
                        3. Give a final response using the gathered information
        """

        react_agent_farm=create_react_agent(client,tools=[get_current_location],state_modifier=SYS_CROP_REC)
        inputs={
            "messages":[
                {"role": "user",
                "content": f"Elaborate on the changes that have happened in landscape patterns"}
            ]
            }
        message=print_stream(react_agent_farm.stream(inputs, stream_mode="values"))
        text=message.content

        return text

@tool
def get_current_location() -> list:
    """This function gives the current location of the user
    returns: city of the user"""
    g = geocoder.ip('me')
    loc = g.address
    return(loc.split(",")[0].strip())

@tool
def get_weather(city:str) -> str:

    """This function gives the current weather of a place in India"""
    api_key = "2a7f14d5bc3df4ecf7e22f21f151344c"

    country_code = "IN"

    current_weather_url = f"https://api.openweathermap.org/data/2.5/weather?q={city},{country_code}&appid={api_key}&units=metric"

    try:
        response = requests.get(current_weather_url,verify=False)
        current_weather_data = response.json()
    except requests.exceptions.RequestException as e:

        print(f"Error fetching current weather data: {e}")

    else:
        print("Current Weather:")
        print(f"Temperature: {current_weather_data['main']['temp']}°C")
        print(f"Description: {current_weather_data['weather'][0]['description']}")
        print(f"Complete Data: {current_weather_data}")

    return current_weather_data

@tool
def get_crop_analysis (imag:str)->str:

    """This tool asks for an image file path of a crop as an input and identifies the possible disease or disease patterns
    noticed in the image"""

    result_img_analysis = predict_crop_disease(imag)

    return(result_img_analysis)

@tool
def get_product_info()-> str:
    """This tool understands the context of the
        crop disease from chat history and then searches the
        internet for possible products to counter the crop disease"""
    tavily_tool = TavilySearchResults(max_results=4)

    SYS_CROP_REC =f"""You are a helpful assistant to a farmer.
                     You are an expert in searching the internet to find the best possible agricultural products
                    Follow these steps:

                    1.Get the current location data using the get_current_location function
                    2.Identify the product the farmer is looking for . Example - Insecticides or Fungicides or Fertilizers etc
                    3.Give your product recommendations from the internet using the tavily_tool which is
                     relevant to the location and farmer's ask
                     4.Give the relevant links as well to see the products
                    """

    react_agent_product=create_react_agent(client,tools=[tavily_tool,get_current_location],state_modifier=SYS_CROP_REC)

    inputs={
    "messages":[
        {"role": "user",
        "content": f"Give the product recommendations as requested"}
    ]
    }
    message=print_stream(react_agent_product.stream(inputs, stream_mode="values"))
    text=message.content

    return text


system_message ="""
You are a helpful assistant to a farmer with an extensive knowledge in agriculture and animal husbandry.

Important instructions to follow:

1.Sequentially call only the specific tools that are mentioned and needed for you to execute the task
2.Take user inputs whenever necessary and do not assume any information. 
3. Give an elaborate final response that has the complete answer (Suggestions/Ideas/Recommendations)and no other question at all.Also
make the structure of the answer look good. 
4.The final AI Message should mandatorily be elaborate and should include all points from the Tool Messages


"""


def text_to_speech(text, target_language_code="hi-IN", speaker="meera", pitch=0, pace=1.65, loudness=1.5, speech_sample_rate=8000, enable_preprocessing=True, model="bulbul:v1"):
  """
  Converts text to speech using the Sarvam.ai API, breaking down long text into chunks.

  Args:
    text (str): The text to convert to speech.
    target_language_code (str, optional): The target language code. Defaults to "hi-IN".
    speaker (str, optional): The speaker voice. Defaults to "meera".
    pitch (int, optional): The pitch of the speech. Defaults to 0.
    pace (float, optional): The pace of the speech. Defaults to 1.65.
    loudness (float, optional): The loudness of the speech. Defaults to 1.5.
    speech_sample_rate (int, optional): The speech sample rate. Defaults to 8000.
    enable_preprocessing (bool, optional): Whether to enable preprocessing. Defaults to True.
    model (str, optional): The model to use. Defaults to "bulbul:v1".

  Returns:
    list of requests.Response: A list of responses from the Sarvam.ai API for each chunk.
  """

  url = "https://api.sarvam.ai/text-to-speech"
  headers = {"Content-Type": "application/json", 'API-Subscription-Key': '5e77cf1b-b61b-4e69-aa07-196835ea3b08'}

  # Break down the text into chunks of maximum 20 characters
  text_chunks = [text[i:i+20] for i in range(0, len(text), 20)]

  responses = []
  for chunk in text_chunks:
    payload = {
        "inputs": [chunk],
        "target_language_code": target_language_code,
        "speaker": speaker,
        "pitch": pitch,
        "pace": pace,
        "loudness": loudness,
        "speech_sample_rate": speech_sample_rate,
        "enable_preprocessing": enable_preprocessing,
        "model": model
    }
    response = requests.request("POST", url, json=payload, headers=headers)
    responses.append(response)

  return responses

def translate_text(text, source_language_code="en-IN", target_language_code="hi-IN", 
                   speaker_gender="Male", mode="formal", model="mayura:v1", 
                   enable_preprocessing=True):
  """
  Translates text using the Sarvam AI API and returns translated text and audio (if available).

  Args:
    text: The text to translate.
    source_language_code: The language code of the source text. Defaults to "en-IN".
    target_language_code: The language code of the target text. Defaults to "hi-IN".
    speaker_gender: The gender of the speaker. Defaults to "Male".
    mode: The translation mode. Defaults to "formal".
    model: The translation model to use. Defaults to "mayura:v1".
    enable_preprocessing: Whether to enable preprocessing. Defaults to True.

  Returns:
    A tuple containing:
      - translated_text: The translated text.
      - audio_data_bytes: The audio data as bytes, if available. Otherwise, None.
  """

  url = "https://api.sarvam.ai/translate"
  payload = {
      "input": text,
      "source_language_code": source_language_code,
      "target_language_code": target_language_code,
      "speaker_gender": speaker_gender,
      "mode": mode,
      "model": model,
      "enable_preprocessing": enable_preprocessing
  }
  headers = {"Content-Type": "application/json", 'API-Subscription-Key': '5e77cf1b-b61b-4e69-aa07-196835ea3b08'}

  response = requests.request("POST", url, json=payload, headers=headers)

  response_json = response.json()

  translated_text = response_json.get("translated_text")

  return translated_text

def beautify_text(text):

    
    client_1 = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = "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"
    )
    completion = client_1.chat.completions.create(
    model="meta/llama-3.1-70b-instruct",
    messages=[{"role":"user","content":f"Beautify the presentation of the text {text}. Give bullet points whereever necessary and presere the integrity of web links wherever applicable"}],
    temperature=0.2,
    top_p=0.7,
    max_tokens=5000,
    stream=True
    )

    b_text = " "

    for chunk in completion:
       
        if chunk.choices[0].delta.content is not None:
            b_text = b_text + chunk.choices[0].delta.content
    print(b_text)
    return b_text


def combine_audio_files(audio_data_list):
  """Combines multiple audio data arrays into one.

  Args:
    audio_data_list: A list of audio data arrays (bytes).

  Returns:
    A tuple containing the combined audio data and its sample rate.
  """

  combined_audio = np.array([])
  first_audio_data = sf.read(io.BytesIO(audio_data_list[0]))[0]
  sample_rate = sf.read(io.BytesIO(audio_data_list[0]))[1] 
  
  for audio_data_bytes in audio_data_list[1:]:
    audio_data = sf.read(io.BytesIO(audio_data_bytes))[0]
    combined_audio = np.concatenate((combined_audio, audio_data), axis=0)

  return combined_audio, sample_rate

user_id="1"
config = {"configurable": {"thread_id": user_id}}
#Creating a cofig stores the chat history with a particular user id.
tavily_tool = TavilySearchResults(max_results=4)
#Create a React agent and provide it with all the tools
router_agent=create_react_agent(client,
                                tools=[get_weather,
                                       get_current_location,
                                        get_crop_analysis,
                                        get_crop_recommendations,
                                        get_product_info,
                                        get_marketing_ideas,
                                        get_farm_state,
                                        tavily_tool
                                    ],
                                state_modifier=system_message,
                                checkpointer=memory,
                                )

def main():
    st.title("AgriBot: Your AI-powered Farming Assistant")

    user_input = st.text_input("Ask AgriBot a question:", key="user_input")
    uploaded_image = st.file_uploader("Upload a crop image (optional):", type=["jpg", "jpeg", "png"])

    if user_input or uploaded_image:
        with st.spinner('AgriBot is thinking...'):
            inputs = {"messages": [("user", user_input)]}

            if uploaded_image:
                bytes_data = uploaded_image.read()
                # Assuming 'bytes_data' is a byte-like object representing the image data
                image_path = 'uploaded_image.jpg'
                with open(image_path, 'wb') as f:
                    f.write(bytes_data)
                inputs["messages"].append(("user", f"Please analyze this image: {image_path}"))

            message=print_stream(router_agent.stream(inputs, config=config,stream_mode="values"))
            
            translated_text  = translate_text(message.content)


            beautified_text = beautify_text(translated_text)

            st.write(beautified_text)

            # Play audio
            responses = text_to_speech(translated_text)
            # st.write(responses)

            audio_data_list = []
            
            for response in responses:
                audio_data_base64 = response.json()["audios"][0]
                # st.write (audio_data_base64)
                audio_data_bytes = base64.b64decode(audio_data_base64)
                audio_data_list.append(audio_data_bytes)

            if audio_data_list:
                combined_audio, sample_rate = combine_audio_files(audio_data_list)
                st.audio(combined_audio, sample_rate=sample_rate)
               
            
            

            # Delete temporary image
            if uploaded_image:
                os.remove(image_path)


if __name__ == "__main__":
    main()