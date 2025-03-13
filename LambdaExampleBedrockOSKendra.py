import boto3 # type: ignore
import os
import json
import traceback
import concurrent.futures
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

# AWS SDK CLIENTS
OSServerless = boto3.client('opensearchserverless')
bedrock = boto3.client(service_name='bedrock-runtime')
kendra = boto3.client('kendra')

#OpenSearch connection
service = 'aoss'
region = 'us-east-1'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key,region, service, session_token=credentials.token)
client = OpenSearch(
  hosts=[{'host': os.environ["OS_Endpoint"], 'port': 443}],
  http_auth=awsauth,
  use_ssl=True,
  verify_certs=True,
  connection_class=RequestsHttpConnection,
  timeout=300
  )
  
Queue = os.environ["QUEUE"]

def search_open_search(index, body):
    return client.search(index=index, body=body)
    
def kendra_search(query):
    index_id = ""
    page_size = 20
    page_number = 1
    # Truncate the query to 1000 characters
    truncated_query = query[:1000]
    
    try:
        return kendra.retrieve(
            IndexId=index_id,
            QueryText=truncated_query,
            PageSize=page_size,
            PageNumber=page_number
        )
    except kendra.exceptions.ValidationException as e:
        print(f"ValidationException: {str(e)}")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def call_bedrockembeddings (text):
    #Bedrock Generate Embeddings
    modelId = 'amazon.titan-embed-text-v1'
    accept = '*/*' #AmazonTitanEmbeddings
    contentType = 'application/json'
    response_bedrock = bedrock.invoke_model(body=text, modelId=modelId, accept=accept, contentType=contentType)
    response_bedrock_body = json.loads(response_bedrock.get('body').read())
    return (response_bedrock_body)
    
def call_bedrockclaude (text, feed, blog, rag_info):
    #modelId = 'anthropic.claude-v2:1'
    modelId = 'anthropic.claude-instant-v1'
    accept = '*/*'
    contentType = 'application/json'

    prompt = f"\n\nHuman: Review the input and provide a few options (please list them in order) to solve the issue.\n"
    prompt += f"\nInputs (might have questions to be addressed): <input>{text}</input>"

    prompt += f"\n<context>\n"
    prompt += f"\n<aws_news>\n"
    for input_tuple in feed:    
        description, link = input_tuple
        prompt += f"<source>{link}</source> <description>{description}</description>\n"
    prompt += f"\n</aws_news>"

    prompt += f"\n<aws_blogposts>\n"
    for input_tuple in blog:    
        description, link = input_tuple
        prompt += f"<source>{link}</source> <description>{description}</description>\n"
    prompt += f"\n</aws_blogposts>"
    
    prompt += f"\n<aws_documentation>\n"
    for input_tuple in rag_info:    
        title, URL, description  = input_tuple
        prompt += f"<website>{URL}</website> <title>{title}</title> <description>{description}</description>\n"
    prompt += f"\n</aws_documentation>\n"
    prompt += f"\n</context>\n"
    
    prompt += f"\n If any news, blog posts, or other information provided as context is relevant, quote and reference the relevant links in your response. Only provide websites included in the <context></context>\n"
    prompt += f"feel free to ask additional questions to solve the customer's questions, and  say 'I dont know' if that's the case"
    prompt += f"\n\nAssistant:"
    
    #print(prompt)
    
    #With Claude instant V1
    body_model = json.dumps({
        #"prompt": f"\n\nHuman:Summarize the following text in less than 2000 tokens in a single text keeping description, use case, functionalities, benefits and best practices to be used by Cloud engineers for understanding if the tool being described might be useful for their use case: {text}\n\nAssistant:",
        "prompt": prompt,
        "max_tokens_to_sample": 2000,
        "temperature": 0.5,
        "top_p": 0.75,
    })
    response_bedrock = bedrock.invoke_model(body=body_model, modelId=modelId, accept=accept, contentType=contentType)
    response_bedrock_body = json.loads(response_bedrock.get('body').read())
    completion_text = '\n'.join(response_bedrock_body['completion'].split('\n')[1:])
    payload = completion_text
    
    print ("Claude response:")
    print(payload)
    print("-----------------")
    return (payload)
    #With Claude instant V1
    
def call_bedrockhaiku (text):    
    prompt = f"\n\nHuman: You are helping to have better understand an AWS workload. Review the input and provide (up to) 3 follow up questions that will be helpful to understand the use case"
    prompt += f"\nInputs: <input>{text}</input>"
    prompt += f"\nDo not include intro phrase or closing remarks, just the 3 questions please."
    ## Cloud Haiku test
    payloadHaiku = {
        "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
        #"modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
        "contentType": "application/json",
        "accept": "application/json",
        "body": json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        })
    }

    # Make the API call to invoke the model
    response_bedrock_Haiku = bedrock.invoke_model(
        body=payloadHaiku['body'], 
        contentType=payloadHaiku['contentType'],
        accept=payloadHaiku['accept'],
        modelId=payloadHaiku['modelId']
    )
    response_bedrock_body = json.loads(response_bedrock_Haiku.get('body').read())
    for item in response_bedrock_body.get('content', []):
        if item.get('type') == 'text':
            #print(item.get('text'))
            response_parsed = item.get('text')

    print ("Haiku response:")
    print(response_parsed)
    print("-----------------")
    return (response_parsed)
    #return (payload)
    ##End Cloud Haiku Test
  
def lambda_handler(event, context):
 feed_description = []
 blog_description = []
 rag_info = []
 #json_payload_body = json.loads(event['body'])
 print (event)
 json_payload_body = json.loads(event["body"])
 print ("received payload")
 print (json_payload_body)
 
 #EMBEDDINGS
 #call Amazon Titan for embeddings on input
 body_bedrock_input = json.dumps({"inputText": json_payload_body["quip_notes"]})
 embedding_body = call_bedrockembeddings (body_bedrock_input)
 embedding_embedding = embedding_body['embedding']
 print ('got the embeddings')

 #OpenSearch Query parameters
 rss_feed_body = {
    'size': 2,
    'query': {'knn': {'rss_feeds': {'vector': embedding_embedding, "k": 2}}}
 }
   
 blog_feed_body = {
    'size': 2,
    'query': {'knn': {'blog_embedding': {'vector': embedding_embedding, "k": 3}}}
 }  

 #SEARCH
 #Concurrent request with threads    
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    future_rss_feed = executor.submit(search_open_search, 'index_feeds', rss_feed_body)
    future_blog_feed = executor.submit(search_open_search, 'blogs_feed', blog_feed_body)
    future_kendra_results = executor.submit(kendra_search, json_payload_body["quip_notes"])

    # Wait for both futures to complete and retrieve results
    results_rss_feed = future_rss_feed.result()
    results_blog_feed = future_blog_feed.result()
    kendra_results = future_kendra_results.result()
            
 print('\nDocuments founded in news feed:')
 print(results_rss_feed)
 for hit in results_rss_feed['hits']['hits']:
  description = hit["_source"]['description']
  link = hit["_source"]['link']
  feed_description.append((description, link))
            
 print('\nDocuments founded in Blogs feed:')
 print(results_blog_feed)
 for hit in results_blog_feed['hits']['hits']:
  description = hit["_source"]['description']
  link = hit["_source"]['link']
  blog_description.append((description, link))

 #Kendra query parse
 for retrieve_result in kendra_results["ResultItems"]:
   print("Passage content: " + str(retrieve_result["Content"]))
   print("Passage URL: " + str(retrieve_result["DocumentURI"]))
   rag_info.append((str(retrieve_result["DocumentTitle"]), str(retrieve_result["DocumentURI"]), str(retrieve_result["Content"])))

#INFERENCE
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    future_bedrockclaude = executor.submit (call_bedrockclaude, json_payload_body["quip_notes"], feed_description, blog_description, rag_info)
    future_bedrockhaiku = executor.submit (call_bedrockhaiku, json_payload_body["quip_notes"])
    
    results_bedrockclaude = future_bedrockclaude.result()
    results_bedrockhaiku = future_bedrockhaiku.result()

#RESPONSE
    
 response = f"{results_bedrockclaude}\n\n *****  Follow up questions ***** \n\n {results_bedrockhaiku}"
 print (response) 
 
 return {
        'statusCode': 200,
        'body': json.dumps(response)
    }