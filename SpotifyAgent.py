import os
import requests
import base64
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional, Literal
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get credentials from environment variables
client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
refresh_token = os.getenv("SPOTIFY_REFRESH_TOKEN")
access_token = os.getenv("SPOTIFY_ACCESS_TOKEN")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)
model = os.getenv("OPENAI_MODEL", "gpt-4o")  # Default to gpt-4o if not specified

#Data models
class PromptRelevance(BaseModel):
    """First LLM call: Extract prompt relevance"""

    description: str = Field(description="Raw user prompt to evaluate")
    is_spotify_request: bool = Field(
        description="Whether this text is related to spotify search or not"
    )
    confidence_score: float = Field(description="Confidence score between 0 and 1")


class SpotifyRequestType(BaseModel):
    """Second LLM call: Extract Spotify request type"""

    description: str = Field(description="Description of the request")
    search_type: Literal["pause","resume","search_and_play","other"] = Field(description="Type of Spotify search request, pause, resume play, search for something and play it, or other")
    confidence_score: float = Field(description="Confidence score between 0 and 1")


class SpotifySearchQuery(BaseModel):
    """Third LLM call if its a search_and_play: Extract Spotify search query"""

    description: str = Field(description="Description of the search query")
    search_query: str = Field(description="Your search query. "
"You can narrow down your search using field filters. The available filters are album, artist, track, year, upc, tag:hipster, tag:new, isrc, and genre. Each field filter only applies to certain result types. "
"The artist and year filters can be used while searching albums, artists and tracks. You can filter on a single year or a range (e.g. 1955-1960). "
"The album filter can be used while searching albums and tracks. "
"The genre filter can be used while searching artists and tracks. "
"The isrc and track filters can be used while searching tracks. "
"The upc, tag:new and tag:hipster filters can only be used while searching albums. The tag:new filter will return albums released in the past two weeks and tag:hipster can be used to return only albums with the lowest 10% popularity. "
"Example: q=remaster%2520track%3ADoxy%2520artist%3AMiles%2520Davis")
    search_type: list[Literal["album", "artist", "playlist", "track", "show", "episode", "audiobook"]] = Field(description="A list of item types to search across. Search results include hits from all the specified item types. For example: ['album', 'track'] returns both albums and tracks matching the query. Allowed values: 'album', 'artist', 'playlist', 'track', 'show', 'episode', 'audiobook'")
    confidence_score: float = Field(description="Confidence score between 0 and 1 indicating how confident the model is that the search query and search type are correct relavant to the user's request")


class PlaybackObject(BaseModel):
    """Playback object for Spotify API"""
    type: Literal["album", "artist", "playlist", "track", "show", "episode", "audiobook"] = Field(description="Type of the playback object")
    uri: str = Field(description="Spotify URI of the playback object")
    description: str = Field(description="Description of the playback object")

#Tools
def extract_prompt_relevance(user_input: str) -> PromptRelevance:
    """First LLM call to determine prompt relevance for Spotify actions"""

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f" Analyze if the text is relevant to the available Spotify actions.",
            },
            {"role": "user", "content": user_input},
        ],
        response_format=PromptRelevance,
    )
    result = completion.choices[0].message.parsed
    return result

def extract_spotify_request_type(description: str) -> SpotifyRequestType:
    """Second LLM call to determine the type of Spotify request"""

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"Extract the type of Spotify request.",
            },
            {"role": "user", "content": description},
        ],
        response_format=SpotifyRequestType,
    )
    result = completion.choices[0].message.parsed
    return result


def extract_spotify_search_query(description: str) -> SpotifySearchQuery:
    """Third LLM call to extract the Spotify search query"""

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"Extract the Spotify search query.",
            },
            {"role": "user", "content": description},
        ],
        response_format=SpotifySearchQuery,
    )
    result = completion.choices[0].message.parsed
    return result


def refresh_spotify_access_token(client_id, client_secret, refresh_token):
    """
    Refreshes the Spotify access token using the refresh token.

    Args:
        client_id (str): Spotify API client ID.
        client_secret (str): Spotify API client secret.
        refresh_token (str): The refresh token obtained during initial authorization.

    Returns:
        str: The new access token, or None if the request fails.
    """
    url = "https://accounts.spotify.com/api/token"
    
    # Encode client_id and client_secret in Base64
    auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    
    # Request headers
    headers = {
        "Authorization": f"Basic {auth_header}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    # Request body
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token
    }
    
    # Make the POST request
    response = requests.post(url, headers=headers, data=data)
    
    if response.status_code == 200:
        # Parse the JSON response
        tokens = response.json()
        print("Refreshed token ", tokens.get("access_token"))
        return tokens.get("access_token")
    else:
        print("Failed to refresh token:", response.status_code, response.text)
        return None

def search_spotify(access_token, query, search_types):
    """
    Searches the Spotify catalog for items matching the query.

    Args:
        access_token (str): The Spotify access token.
        query (str): The search query string (e.g., song name, artist, album).
        search_types (list of str): The types of items to search for. Options include "album", "artist", "playlist", etc.

    Returns:
        dict: The search results as a JSON object, or None if the request fails.
    """
    url = "https://api.spotify.com/v1/search"
    
    # Request headers
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    
    
    # Query parameters
    params = {
        "q": query,
        "type": ",".join(search_types),  # Join the list into a comma-separated string
        "limit": 10  # Limit the number of results (optional)
    }
    
    # Make the GET request
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        # Parse and return the JSON response
        return extract_playback_info(response.json())

    elif response.status_code == 401:
        # Unauthorized, refresh the access token and try again
        print("Refreshing token")
        new_access_token = refresh_spotify_access_token(client_id, client_secret, refresh_token)
        return search_spotify(new_access_token, query, search_types)
    else:
        print("Failed to search Spotify:", response.status_code, response.text)
        return None

def extract_playback_info(spotify_response):
    """Extract the most relevant information from a Spotify search response."""
    
    relevant_info = {
        "primary_results": [],
        "total_results": 0
    }
    
    def add_primary_result(item_type, primary):
        relevant_info["primary_results"].append({
            "type": item_type,
            "name": primary["name"],
            "uri": primary["uri"],
            "id": primary["id"],
            "details": primary
        })
    
    # Handle tracks response
    if "tracks" in spotify_response:
        tracks = spotify_response["tracks"]
        relevant_info["total_results"] += tracks["total"]
        
        valid_items = [item for item in tracks["items"] if item is not None]
        
        if valid_items:
            primary = valid_items[0]
            add_primary_result("track", primary)
    
    # Handle artists response
    if "artists" in spotify_response:
        artists = spotify_response["artists"]
        relevant_info["total_results"] += artists["total"]
        
        valid_items = [item for item in artists["items"] if item is not None]
        
        if valid_items:
            primary = valid_items[0]
            add_primary_result("artist", primary)
    
    # Handle albums response
    if "albums" in spotify_response:
        albums = spotify_response["albums"]
        relevant_info["total_results"] += albums["total"]
        
        valid_items = [item for item in albums["items"] if item is not None]
        
        if valid_items:
            primary = valid_items[0]
            add_primary_result("album", primary)
    
    # Handle playlists response
    if "playlists" in spotify_response:
        playlists = spotify_response["playlists"]
        relevant_info["total_results"] += playlists["total"]
        
        valid_items = [item for item in playlists["items"] if item is not None]
        
        if valid_items:
            primary = valid_items[0]
            add_primary_result("playlist", primary)
    
    # Handle shows response
    if "shows" in spotify_response:
        shows = spotify_response["shows"]
        relevant_info["total_results"] += shows["total"]
        
        valid_items = [item for item in shows["items"] if item is not None]
        
        if valid_items:
            primary = valid_items[0]
            add_primary_result("show", primary)
    
    # Handle episodes response
    if "episodes" in spotify_response:
        episodes = spotify_response["episodes"]
        relevant_info["total_results"] += episodes["total"]
        
        valid_items = [item for item in episodes["items"] if item is not None]
        
        if valid_items:
            primary = valid_items[0]
            add_primary_result("episode", primary)
    
    # Handle audiobooks response
    if "audiobooks" in spotify_response:
        audiobooks = spotify_response["audiobooks"]
        relevant_info["total_results"] += audiobooks["total"]
        
        valid_items = [item for item in audiobooks["items"] if item is not None]
        
        if valid_items:
            primary = valid_items[0]
            add_primary_result("audiobook", primary)
    
    return relevant_info

def pause_spotify(access_token):
    """
    Pauses the current playback on Spotify.

    Args:
        access_token (str): The Spotify access token.

    Returns:
        bool: True if the request is successful, False otherwise.
    """
    url = "https://api.spotify.com/v1/me/player/pause"
    
    # Request headers
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    
    # Make the PUT request
    response = requests.put(url, headers=headers)
    
    if 200 <= response.status_code < 300:
        print("Paused Spotify playback")
        return True
    elif response.status_code == 401:
        # Unauthorized, refresh the access token and try again
        print("Refreshing token")
        new_access_token= refresh_spotify_access_token(client_id, client_secret, refresh_token)
        return pause_spotify(new_access_token)
    else:
        print("Failed to pause Spotify:", response.status_code, response.text)
        return False

def resume_spotify(access_token, uri = None):
    """
    Resumes the current playback on Spotify.

    Args:
        access_token (str): The Spotify access token.

    Returns:
        bool: True if the request is successful, False otherwise.
    """
    url = "https://api.spotify.com/v1/me/player/play"
    
    # Request headers
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    
    # Make the PUT request
    if uri:
        if "album" in uri or "artist" in uri or "playlist" in uri:
            data = {
                "context_uri": uri
            }
            print("Resuming Spotify playback with context URI:", data)
            response = requests.put(url, headers=headers, json=data)
        else:
            data = {
                "uris": [uri]
            }
            print("Resuming Spotify playback with URI:", data)
            response = requests.put(url, headers=headers, json=data)
    else:
        response = requests.put(url, headers=headers)
    
    if 200 <= response.status_code < 300:
        print("Resumed Spotify playback")
        return True
    elif response.status_code == 401:
        # Unauthorized, refresh the access token and try again
        print("Refreshing token")
        new_access_token= refresh_spotify_access_token(client_id, client_secret, refresh_token)
        return resume_spotify(new_access_token)
    else:
        print("Failed to resume Spotify:", response.status_code, response.text)
        return False


def choose_playback_object(search_results, original_query):
    """
    Use OpenAI to choose the most relevant playback object from the search results.

    Args:
        search_results (dict): The search results from the Spotify API.
        original_query (str): The original user query.

    Returns:
        PlaybackObject: The chosen playback object.
    """
    primary_results = search_results.get("primary_results", [])
    
    if not primary_results:
        return None

    # Prepare the input for OpenAI
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that selects the most relevant Spotify playback object based on the user's query."
            },
            {
                "role": "user",
                "content": f"Original query: {original_query}\nPrimary results: {primary_results}"
            }
        ],
        response_format=PlaybackObject,
    )
    
    result = completion.choices[0].message.parsed
    return result

def summarizeFindings(playback_object):
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Summarize the search results in a friendly way that is informative to the user and explains what is now playing."
            },
            {
                "role": "user",
                "content": json.dumps(playback_object.model_dump())
            }
        ],
    )
    result = completion.choices[0].message.content
    print(result)
    

    

#orchestrating function
def spotify_agent(user_input: str):
    # Extract prompt relevance
    prompt_relevance = extract_prompt_relevance(user_input)
    if not prompt_relevance.is_spotify_request or (prompt_relevance.is_spotify_request and prompt_relevance.confidence_score < 0.7):
        return "I'm sorry, I can't help with that."

    print("Prompt relevance:", prompt_relevance.description)
    print("Confidence score:", prompt_relevance.confidence_score)
    
    # Extract Spotify request type
    spotify_request = extract_spotify_request_type(prompt_relevance.description)

    print("Spotify request type:", spotify_request.description)
    print("Search type:", spotify_request.search_type)
    print("Confidence score:", spotify_request.confidence_score)

    if spotify_request.search_type == "other":
        return "I'm sorry, I can't help with that."

    if spotify_request.search_type == "pause":
        print( "I'm pausing the music.")
        return pause_spotify(access_token)
    elif spotify_request.search_type == "resume":
        print( "I'm resuming the music.")
        return resume_spotify(access_token)
    
    
    # Extract Spotify search query
    if spotify_request.search_type == "search_and_play":
        search_query = extract_spotify_search_query(spotify_request.description)

        print("Search query:", search_query.search_query)
        print("Search type:", search_query.search_type)
        print("Confidence score:", search_query.confidence_score)

        search_results = search_spotify(access_token, search_query.search_query, search_query.search_type)
        if search_results:
            for result in search_results["primary_results"]:
                print(f"Type: {result['type']}, Name: {result['name']}")
            playback_object = choose_playback_object(search_results, search_query.search_query)

            print("Playback object:", playback_object)

            resume_spotify(access_token, playback_object.uri)

 
            summarizeFindings(playback_object)
        else:
            return "I couldn't find anything matching your search query."
    
    return "I'm sorry, I can't help with that."


#testing

user_input = input("Enter your request: ")
response = spotify_agent(user_input)
print(response)

#workflow, take in a prompt, give the prompt to the model, 
# the model will first determine if the prompt is asking for something we can do
# so the first tool will return boolean relevant and a confidence interval as well as
# a description, if that's true, we continue, else we return a message saying we can't do that
# second tool, takes the description and a system prompt and we ask it to call the search tool
#passing it a search query and the other params
#make sure to be careful what part of the http response we're adding to the context,
# if we get a 403 we need to call the refresh token tool, add the new 
# token to context and then call the search tool again. 
#if we get a 200 we add parts of the response to the context, 
# then we have a reflection agent that checks the reponse and asks how confident we are that we matched
#what the user is asking for, if we're not confident we ask for more information, if we are confident we
#call the play song tool and pass it the uri of the song, playlist or whatever that we found.
#return the response from the play song tool to the user. with a message.
  