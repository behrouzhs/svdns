# Spectral Word Embedding with Negative Sampling (AAAI 2018)

This software learns a word embedding from the input co-occurrence matrix (preferably extracted from a large corpus such as Wikipedia). This work is published in AAAI 2018. Please refer to the paper for the description of the algorithm. And please don't forget to cite the paper if you use this.

The following instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Requirement before running the software

The Python wrapper automatically compiles the C code and runs it for you. You just need a C compiler to be installed on your system. For instance, in Linux or Mac you need to have gcc compiler and in Windows it uses Visual C++ Command-Line Compiler (cl.exe) which comes with Visual Studio and it can also be installed separately.

In addition to the standard C compiler, you need the following Python libraries:
* Numpy
* Scipy
* Pandas

### A note for Windows users

On Linux, gcc is enough for compiling and no extra configuration is needed. On Windows you may (or may not) need to configure some settings or paths. If you get error messages (related to compiler) while running the Python script, add an environment variable called `INCLUDE` and specify the path to the neccessary include directories that the compiler needs. In addition, add another environment variable called `LIB` and specify the path to the neccessary lib directories that the compiler needs. In my computer they are as follows:

* INCLUDE
`C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\SDK\ScopeCppSDK\SDK\include\ucrt;C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\SDK\ScopeCppSDK\VC\include;C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\Common7\IDE\VC\vcpackages\IntelliSense\iOS\OSS\musl-1.1.10\include;C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\Common7\IDE\VC\vcpackages\IntelliSense\iOS\OSS\musl-1.1.10\arch\x86_64`

* LIB
`C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\SDK\ScopeCppSDK\SDK\lib;C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\SDK\ScopeCppSDK\VC\lib`

## Running the software to train a word embedding

For this purpose, you need to have a large text corpus (e.g Wikipedia) in a single text file. For instance, the latest dump of Wikipedia (articles in XML format) can be downloaded at: [https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2). For a more complete list please visit: [https://dumps.wikimedia.org/enwiki/latest/](https://dumps.wikimedia.org/enwiki/latest/).

These types of corpuses require a lot of preprocessing such as removing HTML tags and structure to get clean text from it, handling or removing special characters, etc. We will not go through the details of preprocessing but it is a neccessary step in order to get a high quality embedding with meaningful and manageable sized vocabulary.

After downloading and extracting the zip file, and also preprocessing steps you will get the clean text file. Let's call the clean text file `corpus_clean.txt`. In order to train and obtain a word embedding, run the following command (example usage):

```
$ python svdns.py -input corpus_clean.txt -output svdns_embedding_d300.txt -mincount 10 -windowsize 10 -dimension 300
```

After running the above command, `svdns_embedding_d300.txt` will be generated which contains the word embeddings. Each row will contain a word and its corresponding vector representation.

## Options and switches for executing the code

* Mandatory parameter:
  * -input \<file\>: Specifies the input corpus file.

* Optional parameters:
  * -vocab \<file\>: Specifies the input vocabulary file. If not specified, a file named "vocab.txt" will be generated. You can specify an existing vocabulary file to be used. This is useful if you want to run and compare multiple algorithms with the exact same vocabulary. \[Default: vocab.txt\]
  * -output \<file\>: Specifies the output embedding file. The resulting word vectors will be stored in this file. \[Default: embedding.txt\]
  * -maxvocab \<int\>: Upper bound on vocabulary size, i.e. keep the \<int\> most frequent words. The minimum frequency words are randomly sampled so as to obtain an even distribution over the alphabet. It is good to specify a limit for the vocabulary size (e.g. 250000) \[Default: 0 (no limit)\]
  * -mincount \<int\>: Lower limit on word frequencies such that words which occur fewer than \<int\> times are discarded. This is very helpful in removing extremely eare words. \[Default: 10\]
  * -windowsize \<int\>: Number of context words to the left and to the right to be considered in the co-occurrence counts. \[Default: 10\]
  * -pmicutoff \<float\>: Filtering threshold to discard unimportant co-occurrences. Using this option will set all the PMI values less than cutoff threshold to zero and the matrix will become sparser. \[Default: -2.5\]
  * -pmishift \<float\>: It will shift all the PMI values by \<float\>. Please note that factorizing the all positive matrix practically yields better embeddings, so try to use `shift=-pmicutoff`. \[Default: 2.5\]
  * -dimension \<int\>: The dimensionality of the word embeddings. \[default: 100\]
  * -engine \<string\>: The engine to use for SVD factorization. Valid options are `c`, `python`, and `auto`. In case of auto, it will use the Python engine if your system has more than 4 cores, otherwise it uses the C engine. For highly multicore systems (cpu_cores >= 8) the Python is faster, but with few cores the C version is faster. This is because the C version is only partially parallelized using OpenMP multithreading library. \[Default: auto\]
  * -thread \<int\>: The number of threads to be used in parallel processing. This option is only used when using the C engine, and it is ignored in Python engine (Python automatically uses multiple cores without your control). \[default: 1\]
  * -verbose \<int\>: Determies the amount of information to be printed in the console. Valid options are 0, 1, and 2. \[Default: 2\]

## Pre-trained word vectors

We have run our algorithm on Wikipedia dump of 20160305 and the pre-trained word vectors file `svdns_wikipedia20160305_d100` which contains the first 100,000 frequent words can be downloaded at the following link. Our trained word vectors contain 163,188 words but because of file size limitation on GitHub we removed the bottom infrequent ones.

[https://github.com/behrouzhs/svdns/raw/master/svdns_wikipedia20160305_d100.zip](https://github.com/behrouzhs/svdns/raw/master/svdns_wikipedia20160305_d100.zip)

## License

[MIT License](https://opensource.org/licenses/MIT)

The algorithm borrows Singular Value Decomposition (SVD) part from SVDLIBC (open source under [BSD License](https://tedlab.mit.edu/~dr/SVDLIBC/license.html)) which is based on ATLAS (open source) linear algebra library.

