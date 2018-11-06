from xml.dom import minidom
import json
import xml.etree.ElementTree as ET
import time
import datetime
import pprint

# article_data = "/home/konstantina/data/semeval/articles-training-20180831.xml";
# ground_truth_data = "/home/konstantina/data/semeval/ground-truth-training-20180831.xml";

# article_data = "/home/konstantina/data/semeval/articles-validation-20180831.xml";
article_data = "/home/konstantina/data/semeval/articles-training-20180831.xml";
# ground_truth_data = "/home/konstantina/data/semeval/ground-truth-validation-20180831.xml";
ground_truth_data = "/home/konstantina/data/semeval/ground-truth-training-20180831.xml";


# get labels
labels_xml = minidom.parse(ground_truth_data);
article_nodes_with_labels = labels_xml.getElementsByTagName('article');
article_label_map = dict();  # from doc id to binary hyperpartisanship
for article_label in article_nodes_with_labels:
    article_label_map[article_label.attributes['id'].value]= article_label.attributes['hyperpartisan'].value;
print("got labels");

# get data
start_time = time.time();
articles_json = dict();  # to transform xml file to json file
articles_json["articles"] = list();
for event, element in ET.iterparse(article_data):
    if element.tag == "article":
        counter = 0;
        for item_tuple in element.items():
            if counter == 0:
                article_id = item_tuple[1];
            elif counter == 1:
                article_date = item_tuple[1];
            elif counter == 2:
                article_title = item_tuple[1];
            counter += 1;
        # decode this bytestring to avoid serialization errors
        article_text = ET.tostring(element, method='text').decode().strip();
        article_text = article_text.replace("\n","");
        # current_article_json = dict();
        # current_article_json["id"] = article_id;
        # current_article_json["published-at"] = article_date;
        # current_article_json["title"] = article_title;
        # current_article_json["text"] = article_text;
        # current_article_json["hyperpartisan"] = article_label_map[article_id];
        # # pprint.pprint(current_article_json);
        # articles_json["articles"].append(current_article_json);
        # if len(articles_json["articles"])%50000 == 0:
        #     print("{} articles in dict".format(len(articles_json["articles"])));
        element.clear();
print("execution time: {}".format(str(datetime.timedelta(seconds=time.time() - start_time))));
print("got data");
# json_file_name = "/home/konstantina/data/semeval/" + \
#                  article_data.split("/")[len(article_data.split("/"))-1] + ".json";
# with open(json_file_name + time.time(), 'w', encoding='utf-8') as outputf:
#     json.dump(articles_json["articles"], outputf, indent=3);
# print("wrote data as json");


