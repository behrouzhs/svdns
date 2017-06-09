# Spectral Word Embedding with Negative Sampling

This software learns a word embedding from the input co-occurrence matrix (preferably extracted from a large corpus such as Wikipedia). This work is submitted to NIPS 2017 and is under review.

The following instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites for input preparation

The input to this algorithm is a word-word co-occurrence matrix. For calculating this co-occurrence matrix, we use existing software from GloVe which can be downloaded at:

* [vocab_count](https://github.com/stanfordnlp/GloVe/blob/master/src/vocab_count.c) - This file is used to scan the corpus and build a vocabulary.
* [cooccur](https://github.com/stanfordnlp/GloVe/blob/master/src/cooccur.c) - This file is used, given a vocabulary, to calculate the word-word co-occurrence matrix.

## Compiling the source code

The source code of the software is written in C and can be compiled using standard C compilers in any operating system (Linux, Windows, and macOS). To compile the prerequisites use:

```
$ gcc -Wall -m64 -O3 vocab_count.c -o vocab_count -lm -lpthread
$ gcc -Wall -m64 -O3 cooccur.c -o cooccur -lm -lpthread
```

You can ignore `-Wall` (show all warnings), `-m64` (compile for 64-bit system), `-O3` (optimization level 3). However, `-lm` (link math library) and `-lpthread` (multi-threading library) are required to compile and run the program.

To compile our program run:

```
$ gcc -Wall -fopenmp -m64 -O3 svdns.c -o svdns -lm
```

Our program uses OpenMP shared memory multi-threading library which is standard and is implemented in almost every C compiler. If you ignore `-fopenmp` switch, it will run on a single thread, however, for better performance use this option.

## Running the software to train a word embedding

For this purpose, you need to have a large text corpus (e.g Wikipedia) in a single text file. For instance, dump of June 1st, 2017 of Wikipedia (articles in XML format) can be downloaded at: [https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2). For the latest dump, you can always refer to [https://dumps.wikimedia.org/enwiki/latest/](https://dumps.wikimedia.org/enwiki/latest/).

These types of corpuses require a lot of preprocessing such as removing HTML tags and structure to get clean text from it, handling or removing special characters, etc. We will not go through the details of preprocessing but it is a neccessary step in order to get a high quality embedding with meaningful and manageable sized vocabulary.

After downloading and extracting the zip file, and also preprocessing steps you will get the clean text file. Let's call the clean text file `corpus_clean.txt`. In order to train and obtain a word embedding, run the following 3 commands one after another:

```
$ ./vocab_count -min-count 5 < ./corpus_clean.txt > ./vocab.txt
$ ./cooccur -window-size 10 -vocab-file ./vocab.txt < ./corpus_clean.txt > ./cooccurrence_matrix.bin
$ ./svdns -pmi2 -pmicutoff -2.5 -shift 2.5 -dimension 100 -thread 8 -vocab ./vocab.txt -input ./cooccurrence_matrix.bin -output ./svdns_embedding_d100.txt
```

After running the above commands, `svdns_embedding_d100.txt` will be generated which contains the word embeddings. Each row will contain a word and its corresponding vector representation.

## Options and switches for executing the code

For `vocab_count` it is good to limit the vocabulary to the words occuring at least `-min-count` times. This option will remove extremely rare words from the vocabulary.

For `cooccur` you need to use a proper `-window-size`. Reasonable range for `-window-size` is between 5 and 15.

For our algorithm `svdns` there are several switches that can be used:

* -pmi2, -pmi10: Base 2 or base 10 Pointwise Mutual Information (PMI) calculation. If not specified, matrix factorization will be done on co-occurrence matrix rather than PMI matrix (which will be a simple SVD on co-occurrences and is not our algorithm).
* -pmicutoff \<float\>: Using this option will set all the PMI values less than cutoff threshold to zero and the matrix will become sparser. (default: -infinity, which does not filter anything.)
* -shift \<float\>: It will shift all the PMI values by a positive or negative value. (default: 0)
* -dimension \<int\>: The dimensionality of the word embedding. (default: 100)
* -thread \<int\>: The number of threads to use in parallel processing. (default: 4)
* -vocab \<file\>: Specifies the input vocabulary file. This vocabulary file is the output of `vocab_count`.
* -input \<file\>: Specifies the input co-occurrence file. This co-occurrence file is the output of `cooccur`.
* -output \<file\>: Specifies the output embedding file. The resulting word vectors will be stored in this file.

## Pre-trained word vectors

We have run our algorithm on Wikipedia dump of 20160305 and the pre-trained word vectors file `svdns_wikipedia20160305_d100` which contains the first 100,000 frequent words can be downloaded at the following link. Our trained word vectors contain 163,188 words but because of file size limitation on GitHub we removed the bottom infrequent ones.

[https://github.com/behrouzhs/svdns/raw/master/svdns_wikipedia20160305_d100.zip](https://github.com/behrouzhs/svdns/raw/master/svdns_wikipedia20160305_d100.zip)

## License

The algorithm borrows Singular Value Decomposition (SVD) part from SVDLIBC (open source) which is based on ATLAS (open source) linear algebra library. This software is provided for the reviewers' attention (NIPS 2017) and it will be licensed under GNU GPL. Redistributions of the software in either source code or binary form is not permitted for now.

