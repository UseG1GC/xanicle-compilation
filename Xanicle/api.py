
curl -X POST https://192.168.1.115:11434/api/tags


curl https://192.168.1.115:11434/api/tags




curl -X POST https://192.168.1.115:11434/api/generate -d '{
  "model": "llama3",
  "prompt":"Why is the sky blue?"
 }'