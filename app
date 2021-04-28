# encoding: utf-8
from flask import Flask, render_template, request, redirect, url_for
import os
import json
from collections import defaultdict, Counter
import random
from utils import *

app = Flask(__name__)

topic_num = 25
data_path = r'static\data'
global_min_year = 2001
global_max_year = 2021
max_topic_p_th = 1 / topic_num * 3

with open(os.path.join(data_path, 'papers.json'), encoding='utf-8') as f:
    all_papers = json.load(f)
with open(os.path.join(data_path, 'topic_descriptors.json'), encoding='utf-8') as f:
    all_topics = json.load(f)

topic_landmarks = [
    {'topic_id': 7, 'avg_x': 0.57, 'avg_y': 5.41},
    {'topic_id': 24, 'avg_x': 4.19, 'avg_y': 2.44},
    {'topic_id': 23, 'avg_x': -1.67, 'avg_y': 3.97},
    {'topic_id': 2, 'avg_x': 1.06, 'avg_y': 2.5},
    {'topic_id': 8, 'avg_x': -0.79, 'avg_y': -1.02},
    {'topic_id': 1, 'avg_x': 5.64, 'avg_y': -4.28},
    {'topic_id': 12, 'avg_x': 0.45, 'avg_y': 0.66},
    {'topic_id': 10, 'avg_x': 4.61, 'avg_y': -0.68},
    {'topic_id': 11, 'avg_x': 1.16, 'avg_y': -1.29},
    {'topic_id': 9, 'avg_x': 6.91, 'avg_y': 2.59},
    {'topic_id': 5, 'avg_x': 8.55, 'avg_y': -0.65},
    {'topic_id': 14, 'avg_x': 4.25, 'avg_y': 5.08},
    {'topic_id': 13, 'avg_x': -2.69, 'avg_y': 0.64},
    {'topic_id': 21, 'avg_x': 3.39, 'avg_y': 2.73},
    {'topic_id': 3, 'avg_x': 2.4, 'avg_y': 2.82},
    {'topic_id': 17, 'avg_x': 3.02, 'avg_y': -2.41},
    {'topic_id': 0, 'avg_x': -0.06, 'avg_y': -3.53},
    {'topic_id': 22, 'avg_x': 3.74, 'avg_y': 1.16},
    {'topic_id': 15, 'avg_x': 4.49, 'avg_y': 1.48},
    {'topic_id': 6, 'avg_x': 1.88, 'avg_y': 1.11},
    {'topic_id': 18, 'avg_x': 3.1, 'avg_y': 0.51},
    {'topic_id': 16, 'avg_x': 5.27, 'avg_y': 0.58},
    {'topic_id': 20, 'avg_x': 2.14, 'avg_y': -4.23},
    {'topic_id': 19, 'avg_x': -2.41, 'avg_y': -2.92},
    {'topic_id': 4, 'avg_x': 2.09, 'avg_y': -0.26}
]


@app.route('/')
def hello_world():
    return redirect(url_for('index'))


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/get_papers_data', methods=['POST'])
def get_papers_data():
    # topic_num = defaultdict(int)
    # topic_x = defaultdict(float)
    # topic_y = defaultdict(float)

    global all_papers
    papers_list = []
    for index, papers in enumerate(all_papers):
        # topic_id = papers['topic_id']
        # topic_num[topic_id] += 1
        # topic_x[topic_id] += papers['x']
        # topic_y[topic_id] += papers['y']
        papers_list.append({
            'id': papers['id'],
            'x': papers['x'],
            'y': papers['y'],
            'topic_id': papers['topic_id'],
            'year': papers['year'],
            'label': papers['label']
        })
    # topic_landmarks = [{'topic_id': topic_id, 'avg_x': round(topic_x[topic_id] / num, 2),
    # 'avg_y': round(topic_y[topic_id] / num, 2)}
    # for topic_id, num in topic_num.items()]
    return json.dumps({'papers_list': papers_list, 'topic_landmarks': topic_landmarks})


@app.route('/get_selected_papers_data', methods=['POST'])
def get_selected_papers_data():
    r = float(request.form['r'])
    cur_min_year = int(request.form['cur_min_year'])
    cur_max_year = int(request.form['cur_max_year'])
    selected_ids = set(json.loads(request.form['selected_ids']))
    top_n = 15
    h, points = find_fine_h(r=r)
    corpus = []
    target_corpus = []
    indexes = set()
    all_indexes = set()

    topics = defaultdict(int)
    years = defaultdict(int)
    journals = defaultdict(int)
    authors = defaultdict(int)

    global all_papers
    for index_, paper in enumerate(all_papers):
        if paper['year'] and cur_min_year <= paper['year'] <= cur_max_year and paper['max_topic_p'] >= max_topic_p_th:
            all_indexes.add(index_)
            if paper['id'] in selected_ids:
                indexes.add(index_)
                journals[paper['journal']] += 1
                topics[paper['topic_id']] += 1
                years[paper['year']] += 1
                for author in paper['author'].split(';'):
                    authors[author] += 1

    journals = [{'journal': x[0], 'n': x[1]} for x in Counter(journals).most_common(top_n) if x[0] != '']
    authors = [x[0] + ' ' + str(x[1]) for x in Counter(authors).most_common(top_n) if x[0] != '']
    topics = [{'topic': x[0], 'n': x[1]} for x in Counter(topics).most_common(top_n)]
    years = [{'year': x[0], 'n': x[1]} for x in Counter(years).most_common(top_n)]

    with open(os.path.join(data_path, 'papers.txt'), encoding='utf-8') as f:
        for index_, line in enumerate(f):
            line = line.strip()
            if index_ in indexes:
                target_corpus.append(line)
                corpus.append(line)
            elif random.uniform(0, 1) < 0.066 and index_ in all_indexes:  # about 1500 papers
                corpus.append(line)

    # line_num + 10 for more
    corpus_counter = Counter(' '.join(corpus).split(' '))
    target_counter = Counter(' '.join(target_corpus).split(' '))
    word2g2 = g2_statistics(corpus_counter, target_counter)
    words = typesetting(word2g2, len(points) + 10, max_length=10)

    return json.dumps({'words': words, 'journals': journals, 'authors': authors, 'topics': topics,
                       'years': years, 'points': points, 'h': h})


@app.route('/get_topic_data', methods=['POST'])
def get_topic_data():
    topic_id = int(request.form['topic_id'])
    return json.dumps(all_topics[topic_id])


if __name__ == '__main__':
    app.run(debug=True)
