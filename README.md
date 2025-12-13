# Project Explanation

Our approach was to combine three different approaches and combine to get the best results. Ultimately, we went with:
1. LoRa fine-tuning - Thomas
2. Retrieval Augmented Generation (RAG) - Alex
3. Prompt Engineering - Chris

## Running Instructions

We have built and included the majority of the necessary files, such as the LoRa's weights and the index used with RAG. However,
checking out either directories should give you a clear explanation as to how you could run them yourselves if you wanted
to change various parameters, especially chunks for the RAG system (and tolerance when adding context to prompts)

To run, first ensure you create a new virtual environment via requirements.txt, then run `python3 example.py` which will give you
a quick example as to how the model works. Be sure to play with the threshold parameter for RAG to include more or less context,
but this is a bare bones example of how you can provide a prompt, and then use the three aformentioned techniques to help write code.