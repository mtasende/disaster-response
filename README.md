# Disaster Response Pipeline Project
A Machine Learning system that uses Natural Language Processing to identify relevant messages in social networks, when a dissaster occurs.

## Installation
To install the project follow the instructions below:
```
$ ./install.sh
```
The script will first preprocess the data and save it into a sqlite database. You can change the database path by modifying the script. Then a model will be trained and saved into pickle format, to be used later, in the web app.

**NOTE: Expect the installation process to take a long time (about one hour may be normal).**

## Project Motivation
When a disaster occurs many messages are sent through different channels (e.g.: News, Social Networks, or directly to responders). It is very important to detect to which responders the messages should arrive (Firefighters, Police, Ambulance, etc.). To that end a quick and accurate classification of those messages is very useful. It is normally not good enough to look for keywords, and that's why a Machine Learning approach could give the solution to this problem.

## File Descriptions
```
├── LICENSE
├── README.md
├── disaster_app            <- Web app main dir
│   └── templates           <- HTML templates for the web app
├── data                    <- Data and data wrangling scripts
├── deploy.sh               <- Script to deploy the app to Heroku
├── disaster_app.py         <- Main script for the web app
├── install.sh              <- Installation script
├── models                  <- ML models and training scripts
├── notebooks               <- Exploratory notebooks
├── requirements.txt
├── run.sh                  <- Script to run the web app
└── setup.py
```

## How to interact with the project
After the installation is complete, run the `run.sh` script. A server will be listening in http://127.0.0.1:5000. Use a browser to access to that address.

**NOTE: Please don't execute `$python run.py` directly.
You can execute `$python disaster_app.py` if you want.**

## Licensing, Authors, Acknowledgements, etc.
Code released under the [MIT](https://github.com/mtasende/airbnb-analysis/blob/master/LICENSE) license.

This project was authored by Miguel Tasende.

It was created as part of Udacity's Data Scientist Nanodegree.
