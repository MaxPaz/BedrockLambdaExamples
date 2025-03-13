import json
import boto3

bedrock = boto3.client(service_name='bedrock-runtime')

#def lambda_handler(event, context):
 
 # phrase = "Hi Andrew, would you like to know how to get more information about AWS? follow the link (aws.amazon.com)"
 
 # body Anthropic
 # body = json.dumps({
 #  "prompt": f"\nHuman:Please provide 5 different options of the following phrase to make it more natural, engaging and fun, keep it within 130 characters (feel free to include emojis and symbols). <phrase> {phrase} </phrase> \n\nAssistant:",
 #  "max_tokens_to_sample": 300,
 #  "temperature": 0.1,
 #  "top_p": 0.9,
 # })

 #body Amazon Titan
 # body = json.dumps({
 #  "inputText": f"\nPlease provide 5 different options of the following phrase to make it more natural, engaging and fun, keep it within 130 characters (feel free to include emojis and symbols). <phrase> {phrase} </phrase>:",
 #   "textGenerationConfig" : {
 #    "maxTokenCount": 8192,
 #    "temperature":0,
 #    "topP":1,
 #   },
 #  }
 # )

 # modelId = 'amazon.titan-text-express-v1'
 # modelId = 'anthropic.claude-instant-v1'
 # accept = '*/*' #AmazonTitanEmbeddings
 # contentType = 'application/json'
 
 # response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
 # response_body = json.loads(response.get('body').read())
 # #print ("completion")
 # #print(response_body.get('completion'))


def get_model(name):
    model = {
        'Sonnet': 'anthropic.claude-3-5-sonnet-20240620-v1:0',
        'Titan': 'amazon.titan-text-express-v1',
        'Instant': 'anthropic.claude-instant-v1'
    }
    return model.get(name)

def lambda_handler(event, context):
 
 phrase = "Hi Andrew, would you like to know how to get more information about AWS? follow the link (aws.amazon.com)"
 
 json_payload_body = json.loads(json.dumps(event))
 print ("-----------------")
 print("model requested by customer: ", json_payload_body["model"])
 model_id = get_model(json_payload_body["model"])
 print ("model_ID: ", model_id)
 print ("-----------------")
 
 messages = [
  {
   "role": "user",
    "content": [
      {
      "text": f"Please provide 5 different options of the following phrase to make it more natural, engaging and fun, keep it within 130 characters (feel free to include emojis and symbols). <phrase> {phrase} </phrase>"
      }
     ]
    }
   ]

 response = bedrock.converse(
  modelId=model_id,
  messages=messages,
  inferenceConfig={
   "maxTokens": 300,
   "temperature": 0.1,
   "topP": 0.9
   }
 )

 output_text = response['output']['message']['content'][0]['text']
 print("Response: ", output_text)

 return {
     'statusCode': 200,
     'body': json.dumps(output_text)
 }