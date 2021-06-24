import nltk
nltk.download('all-corpora')
nltk.download('vader_lexicon')

import subprocess
cmd = ['pip3','install','--user','--upgrade','git+https://github.com/twintproject/twint.git@origin/master#egg=twint']
subprocess.run(cmd)
print("working")