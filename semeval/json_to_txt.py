import json

# tranform the data into txt format, since fasttext uses this particular format for its input data

# article_data = "/home/konstantina/data/semeval/articles-training-20180831.xml";
article_data = "/home/konstantina/data/semeval/articles-validation-20180831.xml";
json_file_name = "/home/konstantina/data/semeval/" + \
                 article_data.split("/")[len(article_data.split("/"))-1] + ".json";
txt_file_name = "/home/konstantina/data/semeval/" + \
                 article_data.split("/")[len(article_data.split("/"))-1] + ".txt";
with open(json_file_name, "r") as inputf:
    article_data_dict = json.load(inputf);
with open(txt_file_name, "w", encoding='utf-8') as outputf:
    for article in article_data_dict:
        outputf.write(article["text"] + "\t" + "__label__" + article["hyperpartisan"] + "\n");
print(len(article_data_dict));