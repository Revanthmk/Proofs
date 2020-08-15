from InstagramAPI import InstagramAPI

InstagramAPI = InstagramAPI("login", "password")
InstagramAPI.login()                       
mediaId = '1469246128528859784_1520706701'  
recipients = []                          
InstagramAPI.direct_share(mediaId, recipients, text='aquest es es darrer')
