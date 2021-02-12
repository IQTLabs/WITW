import os
from flickrapi import FlickrAPI
from pprint import pprint

FLICKR_PUBLIC = os.getenv('FLICKR_PUBLIC_KEY')
FLICKR_SECRET = os.getenv('FLICKR_SECRET')
PRIVACY_FILTER = 1 #only public photos
CONTENT_TYPE = 1 #only photos
HAS_GEO = True  #only geotagged photos
GEO_CTX = 2     #flickr api values looking for outdoor images only

#DC=[[[-77.0990375584,38.8024915206],[-76.9067768162,38.8024915206],[-76.9067768162,39.0012743239],[-77.0990375584,39.0012743239],[-77.0990375584,38.8024915206]]]

flickr = FlickrAPI(FLICKR_PUBLIC, FLICKR_SECRET, format='parsed-json')

extras ='description,license,date_upload,date_taken,original_format,'
extras+='last_update,geo,tags, machine_tags, o_dims, media,'
extras+='url_m,url_n,url_z,url_c,url_l,url_o'

bounds = '-77.0990375584, 38.8024915206, -76.9067768162, 39.0012743239'
city_pics = flickr.photos.search(privacy_filter=PRIVACY_FILTER, bbox=bounds, content_type=CONTENT_TYPE,
has_geo=HAS_GEO, geo_context=GEO_CTX, extras=extras, per_page=100)
photos = city_pics['photos']

pprint(photos)