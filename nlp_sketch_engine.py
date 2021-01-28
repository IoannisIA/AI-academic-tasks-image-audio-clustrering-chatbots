import requests
base_url = 'https://api.sketchengine.eu/bonito/run.cgi'
data = {
 'corpname': 'preloaded/brexit_1', # the corpus to be loaded
 'format': 'json', # return results in JSON format
 # get your API key here: https://app.sketchengine.eu/ in My account
 'username': 'ia', # your registered username here
 'api_key': '128ee29cbe3a4d55ad7021154c52122a' # the API key created
}

for item in ['can', 'may', 'must', 'shall', 'will', 'could', 'might', 'should', 'would']:
 data['q'] = 'q[lemma="' + item + '"]'
 data['fcrit'] = 'word/ 0 lemma/ 0 tag/ 0'
 d = requests.get(base_url + '/freqs', params=data).json()
 print(item, d['Blocks'][0]['totalfrq'])


####################################################################

freq={}
logdice={}
data['lemma'] = 'brexit'
d = requests.get(base_url + '/wsketch', params=data).json()
for i in d['Gramrels'][1]['Words']:
    freq[i['word']] = i['count']
    logdice[i['word']] = i['score']


print(freq)
print(logdice)

max_value = max(freq.values())
max_keys = [k for k, v in freq.items() if v == max_value] # getting all keys containing the `maximum`

print(max_keys, max_value)

max_value = max(logdice.values())
max_keys = [k for k, v in logdice.items() if v == max_value] # getting all keys containing the `maximum`

print(max_keys, max_value)

#################################################################

freq={}
logdice={}
data['lemma'] = 'brexit'
d = requests.get(base_url + '/wsketch', params=data).json()
for i in d['Gramrels'][2]['Words']:
    freq[i['word']] = i['count']
    logdice[i['word']] = i['score']


print(freq)
print(logdice)

max_value = max(freq.values())
max_keys = [k for k, v in freq.items() if v == max_value] # getting all keys containing the `maximum`

print(max_keys, max_value)

max_value = max(logdice.values())
max_keys = [k for k, v in logdice.items() if v == max_value] # getting all keys containing the `maximum`

print(max_keys, max_value)


