#!/bin/bash

rm -rf app_deploy
mkdir app_deploy
cp -r disaster_app app_deploy/
cp README.md app_deploy/
cp LICENSE app_deploy/
cp -r models app_deploy/
cp -r data app_deploy/
cp requirements.txt app_deploy/
cd app_deploy
echo "from disaster_app import app" > disaster_app.py
echo "web gunicorn disaster_app:app" > Procfile
git init
git add .
git commit -m "First commit"
# git commit -m "Updating the app"
# git remote add heroku https://git.heroku.com/diaster-app.git 

heroku login
heroku create disaster-app-mt
git remote -v

git push heroku master

