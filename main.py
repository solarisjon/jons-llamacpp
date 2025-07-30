from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
	filename="mistral-7b-instruct-v0.1.Q5_K_S.gguf"
    )


output = llm(
	"Once upon a time,",
	max_tokens=512,
	echo=True
)

print(output)
