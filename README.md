# Spectral Word Embedding with Negative Sampling (AAAI 2018)

This software learns a word embedding from the input co-occurrence matrix (preferably extracted from a large corpus such as Wikipedia). This work is published in AAAI 2018. Please refer to the paper for the description of the algorithm. And please don't forget to cite the paper if you use this.

The following instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Requirement before running the software

The Python wrapper automatically compiles the C code and runs it for you. You just need a C compiler to be installed on your system. For instance, in Linux or Mac you need to have gcc compiler and in Windows it uses Visual C++ Command-Line Compiler (cl.exe) which comes with Visual Studio and it can also be installed separately.

In addition to the standard C compiler, you need the following Python libraries:
* Numpy
* Scipy
* Pandas

## Running the software to train a word embedding

For this purpose, you need to have a large text corpus (e.g Wikipedia) in a single text file. For instance, the latest dump of Wikipedia (articles in XML format) can be downloaded at: [https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2). For a more complete list please visit: [https://dumps.wikimedia.org/enwiki/latest/](https://dumps.wikimedia.org/enwiki/latest/).

These types of corpuses require a lot of preprocessing such as removing HTML tags and structure to get clean text from it, handling or removing special characters, etc. We will not go through the details of preprocessing but it is a neccessary step in order to get a high quality embedding with meaningful and manageable sized vocabulary.

After downloading and extracting the zip file, and also preprocessing steps you will get the clean text file. Let's call the clean text file `corpus_clean.txt`. In order to train and obtain a word embedding, run the following 3 commands one after another:

```
$ ./vocab_count -min-count 5 < ./corpus_clean.txt > ./vocab.txt
$ ./cooccur -window-size 10 -vocab-file ./vocab.txt < ./corpus_clean.txt > ./cooccurrence_matrix.bin
$ ./svdns -pmicutoff -2.5 -shift 2.5 -dimension 100 -thread 8 -vocab ./vocab.txt -input ./cooccurrence_matrix.bin -output ./svdns_embedding_d100.txt
```

After running the above commands, `svdns_embedding_d100.txt` will be generated which contains the word embeddings. Each row will contain a word and its corresponding vector representation.

## Options and switches for executing the code

For `vocab_count` it is good to limit the vocabulary to the words occuring at least `-min-count` times. This option will remove extremely rare words from the vocabulary.

For `cooccur` you need to use a proper `-window-size`. Reasonable range for `-window-size` is between 5 and 15.

For our algorithm `svdns` there are several switches that can be used:

* Mandatory parameters:
  * -input \<file\>: Specifies the input co-occurrence file. This co-occurrence file is the output of `cooccur`.
  * -vocab \<file\>: Specifies the input vocabulary file. This vocabulary file is the output of `vocab_count`.
  * -output \<file\>: Specifies the output embedding file. The resulting word vectors will be stored in this file.
  
* Optional parameters:
  * -pmicutoff \<float\>: Using this option will set all the PMI values less than cutoff threshold to zero and the matrix will become sparser. (default: -2.5)
  * -shift \<float\>: It will shift all the PMI values by \<float\> (default: 2.5). Please note that factorizing the all positive matrix practically yields better embeddings, so try to use `shift=-pmicutoff`
  * -dimension \<int\>: The dimensionality of the word embedding. (default: 100)
  * -thread \<int\>: The number of threads to use in parallel processing. (default: 4)
  * -pmi10: calculate Pointwise Mutual Information (PMI) using base 10 logarithm. If not specified, it will use log2 by default.

## Pre-trained word vectors

We have run our algorithm on Wikipedia dump of 20160305 and the pre-trained word vectors file `svdns_wikipedia20160305_d100` which contains the first 100,000 frequent words can be downloaded at the following link. Our trained word vectors contain 163,188 words but because of file size limitation on GitHub we removed the bottom infrequent ones.

[https://github.com/behrouzhs/svdns/raw/master/svdns_wikipedia20160305_d100.zip](https://github.com/behrouzhs/svdns/raw/master/svdns_wikipedia20160305_d100.zip)

## License

[MIT License](https://opensource.org/licenses/MIT)

The algorithm borrows Singular Value Decomposition (SVD) part from SVDLIBC (open source under [BSD License](https://tedlab.mit.edu/~dr/SVDLIBC/license.html)) which is based on ATLAS (open source) linear algebra library.

