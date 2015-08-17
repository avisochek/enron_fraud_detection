#!/usr/bin/python

#from nltk.stem.snowball import SnowballStemmer
import string
    
def parseOutText(f):
    """ given an opened email file f, parse out all text below the
    metadata block at the top
    (in Part 2, you will also add stemming capabilities)
    and return a string that contains all the words
    in the email (space-separated) 
    
    example use case:
    f = open("email_file_name.txt", "r")
    text = parseOutText(f)
    
    """
    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()
    ### split off metadata
    content = all_text.split("X-FileName:")
#    words = ""
#    if len(content) > 1:
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)
#        print text_string
        #stemmer = SnowballStemmer("english")
#        splitted_text = [x for x in text_string.split() if x != ""]
#        words = ''
#        for word in splitted_text:
            #print word
            #word = stemmer.stem(word)
#            words += word+' '
    return text_string #words

def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print text
if __name__ == '__main__':
    main()
