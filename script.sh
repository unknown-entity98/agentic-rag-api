#!/bin/bash

echo "Starting FastAPI server..."
uvicorn main:app &
SERVER_PID=$!

# TODO: wait for /health to return 200
# hint: loop with curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health
# keep checking every 1 second until you get "200", with a max retry count

STATUS = $(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
echo $STATUS

echo "Server is ready."

# TODO: run a test upload using curl, capture the response

# TODO: run a test ask using curl, capture the response

# TODO: kill the server
kill $SERVER_PID
