import re


emoji=re.compile("&#[0-9]*")
amp=re.compile("&amp")
name=re.compile("@[a-z,A-Z,0-9,_]*")
http_link=re.compile("http://t.co[a-z,A-Z,/,0-9]*")
token=re.compile("[,.;<>:]")

x="you's , . a muthaf***in lie &#8220;@LifeAsKing: @20_Pearls @corey_emanuel right! His TL is trash &#8230;. Now, mine? Bible scriptures and hymns&#8221;"
import enchant
from enchant.tokenize import get_tokenizer
x=re.sub(emoji,'',x).lower().strip()
x=re.sub(name,'',x)
x = re.sub(http_link, '', x)
x = re.sub(amp, '', x)
x = re.sub(token, '', x)
print(x)
dd=enchant.Dict("en_US")
after=[]
for c in x.split(' '):
    if c.strip().__len__()==0:
        continue
    if not dd.check(c):
        print('1',c,dd.suggest(c))
        a_list=dd.suggest(c)
        if a_list.__len__():
            after.append(dd.suggest(c)[0])
    else:
        after.append(c)
print(' '.join(after))
print(x)