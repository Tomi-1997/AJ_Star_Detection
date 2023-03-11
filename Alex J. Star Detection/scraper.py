import json
import os
import urllib.request


mydir = 'D:\\UNI\\FinalProject\\scrape\\'
# filename = 'acSearch_Results.json'


def remove_useless_keys(d: dict):
    useless_keys = ['title', 'price', 'date', 'last']
    for k in useless_keys:
        if k in d.keys():
            del d[k]


def remove_ads(d: dict):
    if d['id'] == 'ad':
        d.clear()

# basically takes the relevant details from description to label the data 
def json_cleaner(name: str):
    save_path = "D:\\UNI\\FinalProject\\scrape\\images\\"
    files = [f for f in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, f))]
    files = [f.split('.')[0] for f in files]

    new_data = {}
    dict_list = []
    with open(mydir + name) as f:
        data = json.load(f)
    for i in data['acSearch']:
        temp_dict = {}
        irr_id = str(i['id'])
        if irr_id in files:
            temp_dict['id'] = irr_id
            desc = i['description'].lower()
            # the star rayes labeling
            if ('eight' in desc) or ('achtstrahliger' in desc): # for german desc (didn't have german in six)
                temp_dict['rayes'] = 8
            elif 'six' in desc:
                temp_dict['rayes'] = 6
            else:
                temp_dict['rayes'] = 0
            # the diadem labeling
            if 'diadem' in desc:
                temp_dict['diadem'] = 1
            else:
                temp_dict['diadem'] = 0
        if temp_dict != {}:
            dict_list.append(temp_dict)

    new_data['data'] = dict_list

    with open(mydir + 'new_' + name, 'w') as f:
        json.dump(new_data, f, indent=4)


# def clean_irrelevant():
#     save_path = "D:\\UNI\\Year03\\FinalProject\\scrape\\images\\"
#     files = [f for f in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, f))]
#     files = [f.split('.')[0] for f in files]
#
#     with open(mydir + 'acSearch_Results.json') as f:
#         data = json.load(f)
#
#     for i in data['acSearch']:
#         remove_useless_keys(i)
#         flag = 0
#         irr_id = str(i['id'])
#         for f in files:
#             if f == irr_id:
#                 flag += 1
#         if flag == 0:
#             print("cleared")
#             i.clear()
#             data['acSearch'].remove(i)
#
#     with open(mydir + 'new_' + 'new_acSearch_Results.json', 'w') as f:
#         json.dump(data, f, indent=4)


def process(filename: str = None, filepath: str = None) -> None:
    save_path = "D:\\UNI\\FinalProject\\scrape\\images\\" + filename + ".jpg"

    urllib.request.urlretrieve(filepath, save_path)


def scrape_by_json(filename: str):
    images = []
    site_dir = "https://www.acsearch.info/"
    with open(mydir + filename) as f:
        data = json.load(f)
    for i in data['acSearch']:
        tup = (i['id'], site_dir + i['image'])
        images.append(tup)

    for img in images:
        process(img[0], img[1])


if __name__ == '__main__':
    # if 'eight' in '644. Obverse: Eight-rayed sta'.lower():
    #     print('hi')
    scrape_by_json("results_new.json")
    json_cleaner("results_new.json")
    # clean_irrelevant()
