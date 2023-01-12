import urllib.request
import json
# 모듈 추가

client_id = "gBEK1DmpwzJhcM4rf172" #애플리케이션 등록시 발급 받은 값 입력
client_secret = "7PnzjXXu6O" #애플리케이션 등록시 발급 받은 값 입력
# 클라이언트 id&secret 저장



class NaverDictionaryParsing():
    def __init__(self) -> None:
        pass
    
    def getThumnail(self, word):
        encUrl = "https://openapi.naver.com/v1/search/encyc"
        encQuery = "?query=" + urllib.parse.quote(word)
        encOption = "&display=1&sort=count"
        url = encUrl + encQuery + encOption
        # request url 생성

        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id",client_id)
        request.add_header("X-Naver-Client-Secret",client_secret)
        # request 객체 생성 및 header에 id&secret 추가

        response = urllib.request.urlopen(request)
        # urlopen 및 response 획득
        
        rescode = response.getcode()
        if(rescode==200):
            json_rt = response.read().decode('utf-8')
            py_rt = json.loads(json_rt)
            items = py_rt['items']
            return items[0]['thumbnail']
        else:
            return "Error Code:" + rescode
  
# HTTP status code 확인하여 response 데이터 디코딩


