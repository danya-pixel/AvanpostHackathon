from requests import post



#TODO: move it to os.env
JWT_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3MDE1NjE2NDgsInVzZXJfaWQiOjk4OTg5ODk4OTg5fQ.9Zxsjmyv2KvakIVPlkXLjD8OMt0hRqbXtqjnFohFPY8"

def upload( file):
    files={'file': (str(file.name), file.getvalue(), 'application/zip')}
    headers={
        "Authorization": f"Bearer {JWT_TOKEN}"
    }
    r = post("https://fds.es.nsu.ru/upload/", files=files, headers=headers)
    if r.status_code != 200:
        raise ValueError("Upload error")    
    return "https://fds.es.nsu.ru/uploads/" + r.json()['file_id']
