/opt/mpich/bin/mpic++  pagerank.cpp  -O3 -m64 --force-addr  -ftree-vectorize -msse2 -ftree-vectorizer-verbose=1  -fopenmp   -L /usr/local/lib -std=c++11  -o pagerank -lzmq -lsnappy -lz -lpthread -lglog
/opt/mpich/bin/mpic++  pagerank.cpp  -O3 -m64 --force-addr  -ftree-vectorize -msse2 -ftree-vectorizer-verbose=1  -fopenmp   -L /usr/local/lib -std=c++11  -o pagerank -lzmq -lsnappy -lz -lpthread -lglog
/opt/mpich/bin/mpic++  sssp.cpp  -O3 -m64 --force-addr  -ftree-vectorize -msse2 -ftree-vectorizer-verbose=1  -fopenmp   -L /usr/local/lib -std=c++11  -o sssp -lzmq -lsnappy -lz -lpthread -lglog
#/opt/mpich/bin/mpic++  cc.cpp  -O2    -L /usr/local/lib -std=c++11  -o cc -lzmq -lsnappy -lz -lpthread -lglog
#/opt/mpich/bin/mpic++  sssp.cpp  -O2  -L /usr/local/lib -std=c++11  -o sssp -lzmq -lsnappy -lz -lpthread -lglog
#/opt/mpich/bin/mpic++  pagerank.cpp  -O2  -L /usr/local/lib -std=c++11  -o pagerank -lzmq -lsnappy -lz -lpthread -lglog
