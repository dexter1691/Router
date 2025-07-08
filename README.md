
# Task:
Your task is to create a system that allows Sky to minimize the list of tools sent to Sky in its first request. More formally, given a user query, the system should return the set of integrations required to achieve the task. The list of integrations and their tools are described below. 

```
Calendar
	create_event
	update_event
	delete_event
	search_events
Files
	create_file
	update_file
	trash_file
FaceTime
	start_call
Mail
	send_email
	search_inbox
Maps
	get_directions
	search_location
Music
	play_song
	search_library
Messages
	send_message
	search_messages
Notes
	create_note
	update_note
Reminders
	create_reminder
	update_reminder
Slack
	send_message
System
	capture_screenshot
Stocks
  find_symbol
WebBrowser
  open_url
Weather
	get_weather
```


For the example above, the desirable system output would be the set [Calendar, Messages].

## Strategies:

1. **Use a RAG / Cosine Similarity Model:**  
Use a CLIP style model to measure the similarity of the query with the tool that we need to use. This will 
probably work well because we can cache the embeddings of the tool, and just run a cosine similarity against the query. Building an embedding should be quite fast.  

2. **Use a LLM model to directly output a tool given the query.**  
This should work too, but this has a problem that
to generalize to new tools / fewer tools, we will need to create a bigger dataset that might be harder than sentence similarity. But this allows the model to "reason" in natural langauge which is a nice property to have.   



# TODOs:
- Generalization to new tools (train on subset. test on unseen subset.)
- 