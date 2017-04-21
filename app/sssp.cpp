/*
 ============================================================================
 Name        : GraphPS.c
 Author      : Sun Peng
 Version     :
 Copyright   : Your copyright notice
 Description : Compute Pi in MPI C++
 ============================================================================
 */

#include "../system/GraphPS.h"

using namespace std;

template<class T>
bool comp_sssp(const int32_t P_ID,
               std::string DataPath,
               const int32_t VertexNum,
               T* VertexValue,
               T* VertexMsg,
               T* VertexMsgNew,
               const int32_t* _VertexOut,
               const int32_t* _VertexIn,
               const int32_t step) {
  int32_t *EdgeData, *indices, *indptr;
  int32_t start_id, end_id;
  std::vector<T> result;
  init_comp<T>(P_ID, DataPath, &EdgeData, &start_id, &end_id, &indices, &indptr, std::ref(result));
  // LOG(INFO) << end_id << " " << start_id;
  int32_t i   = 0;
  int32_t j   = 0;
  T   min = 0;
  int32_t changed_num = 0;
  T tmp;
  for (i = 0; i < end_id-start_id; i++) {
    min = VertexMsg[start_id+i];
    for (j = 0; j < indptr[i+1] - indptr[i]; j++) {
      tmp = VertexMsg[indices[indptr[i] + j]] + 1;
      if (min > tmp)
        min = tmp;
    }
    result[i] = min;
#ifdef USE_ASYNC
    VertexMsg[start_id+i] = min;
#endif
    if (min != VertexMsg[start_id+i]) {
      changed_num++;
    }
    if (min != VertexValue[start_id+i]) {
      _Changed_Vertex++;
      VertexValue[start_id+i] = min;
    }
  }
  end_comp<T>(P_ID, EdgeData, start_id, end_id, changed_num, VertexMsg, VertexMsgNew, std::ref(result));
  return true;
}

template<class T>
class PagerankPS : public GraphPS<T> {
public:
  PagerankPS():GraphPS<T>() {
    this->_comp = comp_sssp<T>;
  }
  void init_vertex() {
    this->_VertexValue.assign(this->_VertexNum, 0);
    this->_VertexMsg.assign(this->_VertexNum, GPS_INF);
    this->_VertexMsg[1] = 0;
  }
};

int main(int argc, char *argv[]) {
  start_time_app();
  FLAGS_logtostderr = 1;
  google::InitGoogleLogging(argv[0]);
  init_workers();
  PagerankPS<int32_t> pg;
  //PagerankPS<float> pg;
  // Data Path, VertexNum number, Partition number, thread number, Max Iteration
  // pg.init("/home/mapred/GraphData/eu/edge/", 1070560000, 5096,  2000);
  // pg.init("/home/mapred/GraphData/twitter/edge2/", 41652250, 294,  2000);
  // pg.init("/home/mapred/GraphData/uk/edge3/", 787803000, 2379,  2000);
  pg.init("/home/mapred/GraphData/webuk_3/", 133633040, 300,  2000);
  pg.run();
  stop_time_app();
  LOG(INFO) << "Used " << APP_TIME/1000.0 << " s";
  finalize_workers();
  return 0;
}

