from xml.dom import minidom
import json
import time
import datetime

# article_data = "test_article.xml";
article_data = "/home/konstantina/data/semeval/articles-training-20180831.xml";
# ground_truth_data = "test_article.ground_truth.xml";
ground_truth_data = "/home/konstantina/data/semeval/ground-truth-training-20180831.xml";

def get_node_text(nodelist):
    rc = [];
    for node in nodelist:
        # from documentation
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data);
        else:
            # recursive
            rc.append(get_node_text(node.childNodes));
    return ''.join(rc);

start_time = time.time();
articles_xml = minidom.parse(article_data, bufsize=None); # takes forever for big file
# tree = etree.parse(article_data); # this as well
print(str(datetime.timedelta(seconds=time.time() - start_time)));
article_nodes = articles_xml.getElementsByTagName('article');
print("{} articles in this file".format(len(article_nodes)));
labels_xml = minidom.parse(ground_truth_data);
article_nodes_with_labels = labels_xml.getElementsByTagName('article');
article_label_map = dict();  # from doc id to binary hyperpartisanship
for article_label in article_nodes_with_labels:
    article_label_map[article_label.attributes['id'].value]= article_label.attributes['hyperpartisan'].value;
articles_json = dict();  # to transform xml file to json file
articles_json["articles"] = list();
for a in article_nodes:
    article_id = a.attributes['id'].value;
    article_title = a.attributes['title'].value;
    article_date = a.attributes['published-at'].value;
    # print("id={}, published-at={}, title={}".format(article_id, article_date, article_title));
    paragraphs = a.getElementsByTagName('p');
    # print("{} paragraphs".format(len(paragraphs)));
    article_text = "";
    for p in paragraphs:
        article_text += get_node_text(p.childNodes);
    # print("Article:\n{}".format(article_text));
    # print("Hyperpartisan:\n{}".format(article_label_map[a.attributes['id'].value]));
    current_article_json = dict();
    current_article_json["id"] = article_id;
    current_article_json["published-at"] = article_date;
    current_article_json["title"] = article_title;
    current_article_json["text"] = article_text;
    current_article_json["hyperpartisan"] = article_label_map[a.attributes['id'].value];
    articles_json["articles"].append(current_article_json);
    if len(articles_json)%1000 == 0:
        print("{} articles in dict".format(len(articles_json)));
# print(articles_json);
with open("/home/konstantina/data/semeval/" +
                  article_data.split("/")[len(article_data.split("/"))-1] + ".json", 'wb') as outputf:
    json.dump(articles_json, outputf);

