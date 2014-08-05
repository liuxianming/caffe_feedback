// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_OPTIMIZATION_FEEDBACK_SOLVER_HPP_
#define CAFFE_OPTIMIZATION_FEEDBACK_SOLVER_HPP_

#include <string>
#include <vector>

namespace caffe {

template <typename Dtype>
class FeedbackSolver {
 public:
  explicit FeedbackSolver(const SolverParameter& param);
  explicit FeedbackSolver(const string& param_file);
  void Init(const SolverParameter& param);
  // The main entry of the solver function. In default, iter will be zero. Pass
  // in a non-zero iter number to resume training for a pre-trained net.
  virtual void Solve(const char* resume_file = NULL);
  inline void Solve(const string resume_file) { Solve(resume_file.c_str()); }
  virtual ~FeedbackSolver() {}
  inline shared_ptr<FeedbackNet<Dtype> > net() { return net_; }

 protected:
  // PreSolve is run before any solving iteration starts, allowing one to
  // put up some scaffold.
  virtual void PreSolve() {}
  // Get the update value for the current iteration.
  virtual void ComputeUpdateValue() = 0;
  // The Solver::Snapshot function implements the basic snapshotting utility
  // that stores the learned net. You should implement the SnapshotSolverState()
  // function that produces a SolverState protocol buffer that needs to be
  // written to disk together with the learned net.
  void Snapshot();
  // The test routine
  void Test();
  virtual void SnapshotSolverState(SolverState* state) = 0;
  // The Restore function implements how one should restore the solver to a
  // previously snapshotted state. You should implement the RestoreSolverState()
  // function that restores the state from a SolverState protocol buffer.
  void Restore(const char* resume_file);
  virtual void RestoreSolverState(const SolverState& state) = 0;

  SolverParameter param_;
  int iter_;
  shared_ptr<FeedbackNet<Dtype> > net_;
  shared_ptr<FeedbackNet<Dtype> > test_net_;

  DISABLE_COPY_AND_ASSIGN(FeedbackSolver);
};


template <typename Dtype>
class SGDFeedbackSolver : public FeedbackSolver<Dtype> {
 public:
  explicit SGDFeedbackSolver(const SolverParameter& param)
      : FeedbackSolver<Dtype>(param) {}
  explicit SGDFeedbackSolver(const string& param_file)
      : FeedbackSolver<Dtype>(param_file) {}

 protected:
  virtual void PreSolve();
  Dtype GetLearningRate();
  virtual void ComputeUpdateValue();
  virtual void SnapshotSolverState(SolverState * state);
  virtual void RestoreSolverState(const SolverState& state);
  // history maintains the historical momentum data.
  vector<shared_ptr<Blob<Dtype> > > history_;

  DISABLE_COPY_AND_ASSIGN(SGDFeedbackSolver);
};


}  // namespace caffe

#endif  // CAFFE_OPTIMIZATION_FEEDBACK_SOLVER_HPP_
