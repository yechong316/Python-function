music_media = ['compact disc', '8-track tape', 'long playing record']
new_media = ['DVD Audio disc', 'Super Audio CD']



music_media.append(new_media)
print('append:', music_media)


music_media = ['compact disc', '8-track tape', 'long playing record']
new_media = ['DVD Audio disc', 'Super Audio CD']

music_media.extend(new_media)
print('extend:', music_media)