from InstagramAPI import InstagramAPI 

username = 'your_username_here'
password = 'your_password_here'

ig = InstagramAPI(username, password) 

ig.login() 

ig.getSelfUserFeed()

MediaList = ig.LastJson 

Media = MediaList['items'][0]

MediaID = Media['id']
MediaType = Media['media_type']

isDeleted = ig.deleteMedia(MediaID, media_type=MediaType)

if isDeleted:
    print("Your Media {0} has been deleted".format(
        MediaID
    ))
else:
    print("Your Media Not Deleted")
