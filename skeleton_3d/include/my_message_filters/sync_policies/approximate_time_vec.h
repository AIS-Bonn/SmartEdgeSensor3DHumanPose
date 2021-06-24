/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

#ifndef MESSAGE_FILTERS_SYNC_APPROXIMATE_TIME_VEC_H
#define MESSAGE_FILTERS_SYNC_APPROXIMATE_TIME_VEC_H

#include "my_message_filters/synchronizer_vec.h"

#include <message_filters/connection.h>
#include <message_filters/null_types.h>

#include <boost/tuple/tuple.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
#include <boost/thread/mutex.hpp>

#include <boost/bind.hpp>

#include <ros/assert.h>
#include <ros/message_traits.h>
#include <ros/message_event.h>

#include <deque>
#include <vector>
#include <string>

namespace message_filters
{
namespace sync_policies
{

template<typename M>
struct ApproximateTimeVec
{
  typedef SynchronizerVec<ApproximateTimeVec> Sync;
  typedef M Message;
  typedef typename ros::MessageEvent<M const> Event;
  typedef std::deque<Event> Deque;
  typedef std::vector<Event> Vector;
  typedef std::vector<Event> Tuple;
  typedef std::vector<Deque> DequeTuple;
  typedef std::vector<Vector> VectorTuple;

  ApproximateTimeVec(uint32_t queue_size, int num_msgs)
  : parent_(0)
  , queue_size_(queue_size)
  , num_msgs_(num_msgs)
  , NO_PIVOT(num_msgs)
  , deques_(num_msgs)
  , num_non_empty_deques_(0)
  , past_(num_msgs)
  , candidate_(num_msgs)
  , pivot_(NO_PIVOT)
  , max_interval_duration_(ros::DURATION_MAX)
  , age_penalty_(0.1)
  , has_dropped_messages_(num_msgs, false)
  , inter_message_lower_bounds_(num_msgs, ros::Duration(0))
  , warned_about_incorrect_bound_(num_msgs, false)
  {
    ROS_ASSERT(queue_size_ > 0);  // The synchronizer will tend to drop many messages with a queue size of 1. At least 2 is recommended.
  }

  ApproximateTimeVec(const ApproximateTimeVec& e)
  {
    *this = e;
  }

  ApproximateTimeVec& operator=(const ApproximateTimeVec& rhs)
  {
    parent_ = rhs.parent_;
    queue_size_ = rhs.queue_size_;
    num_msgs_ = rhs.num_msgs_;
    NO_PIVOT = rhs.NO_PIVOT;
    num_non_empty_deques_ = rhs.num_non_empty_deques_;
    pivot_time_ = rhs.pivot_time_;
    pivot_ = rhs.pivot_;
    max_interval_duration_ = rhs.max_interval_duration_;
    age_penalty_ = rhs.age_penalty_;
    candidate_start_ = rhs.candidate_start_;
    candidate_end_ = rhs.candidate_end_;
    deques_ = rhs.deques_;
    past_ = rhs.past_;
    has_dropped_messages_ = rhs.has_dropped_messages_;
    inter_message_lower_bounds_ = rhs.inter_message_lower_bounds_;
    warned_about_incorrect_bound_ = rhs.warned_about_incorrect_bound_;

    return *this;
  }

  void initParent(Sync* parent)
  {
    parent_ = parent;
    assert(parent_->get_num_msgs() == num_msgs_);
  }

  void checkInterMessageBound(int i)
  {
    namespace mt = ros::message_traits;
    if (warned_about_incorrect_bound_[i])
    {
      return;
    }
    Deque& deque = deques_[i];
    Vector& v = past_[i];
    ROS_ASSERT(!deque.empty());
    const Message &msg = *(deque.back()).getMessage();
    ros::Time msg_time = mt::TimeStamp<Message>::value(msg);
    ros::Time previous_msg_time;
    if (deque.size() == (size_t) 1)
    {
      if (v.empty())
      {
        // We have already published (or have never received) the previous message, we cannot check the bound
        return;
      }
      const Message &previous_msg = *(v.back()).getMessage();
      previous_msg_time = mt::TimeStamp<Message>::value(previous_msg);
    }
    else
    {
      // There are at least 2 elements in the deque. Check that the gap respects the bound if it was provided.
      const Message &previous_msg = *(deque[deque.size()-2]).getMessage();
      previous_msg_time =  mt::TimeStamp<Message>::value(previous_msg);
    }
    if (msg_time < previous_msg_time)
    {
      ROS_WARN_STREAM("Messages of type " << i << " arrived out of order (will print only once)");
      warned_about_incorrect_bound_[i] = true;
    }
    else if ((msg_time - previous_msg_time) < inter_message_lower_bounds_[i])
    {
      ROS_WARN_STREAM("Messages of type " << i << " arrived closer (" << (msg_time - previous_msg_time)
		      << ") than the lower bound you provided (" << inter_message_lower_bounds_[i]
		      << ") (will print only once)");
      warned_about_incorrect_bound_[i] = true;
    }
  }

  void add(const Event& evt, int i)
  {
    //ROS_INFO("Approx time add event %d", i);
    boost::mutex::scoped_lock lock(data_mutex_);

    Deque& deque = deques_[i];
    deque.push_back(evt);
    if (deque.size() == (size_t)1) {
      // We have just added the first message, so it was empty before
      ++num_non_empty_deques_;
      if (num_non_empty_deques_ == num_msgs_)
      {
        // All deques have messages
        //ROS_INFO("Approx. time process (1) !");
        process();
      }
    }
    else
    {
      checkInterMessageBound(i);
    }
    // Check whether we have more messages than allowed in the queue.
    // Note that during the above call to process(), queue i may contain queue_size_+1 messages.
    Vector& past = past_[i];
    if (deque.size() + past.size() > queue_size_)
    {
      // Cancel ongoing candidate search, if any:
      num_non_empty_deques_ = 0; // We will recompute it from scratch
      for (int j = 0; j < num_msgs_; ++j) {
        recover(j);
      }
      // Drop the oldest message in the offending topic
      ROS_ASSERT(!deque.empty());
      deque.pop_front();
      has_dropped_messages_[i] = true;
      if (pivot_ != NO_PIVOT)
      {
        // The candidate is no longer valid. Destroy it.
        candidate_ = Tuple(num_msgs_);
        pivot_ = NO_PIVOT;
        // There might still be enough messages to create a new candidate:
        //ROS_INFO("Approx. time process (2) !");
        process();
      }
    }

    //ROS_INFO("... done Approx time add");
  }

  void setAgePenalty(double age_penalty)
  {
    // For correctness we only need age_penalty > -1.0, but most likely a negative age_penalty is a mistake.
    ROS_ASSERT(age_penalty >= 0);
    age_penalty_ = age_penalty;
  }

  void setInterMessageLowerBound(int i, ros::Duration lower_bound) {
    ROS_ASSERT(lower_bound >= ros::Duration(0,0));
    inter_message_lower_bounds_[i] = lower_bound;
  }

  void setInterMessageLowerBound(ros::Duration lower_bound) {
    ROS_ASSERT(lower_bound >= ros::Duration(0,0));
    for (size_t i = 0; i < inter_message_lower_bounds_.size(); i++)
    {
      inter_message_lower_bounds_[i] = lower_bound;
    }
  }

  void setMaxIntervalDuration(ros::Duration max_interval_duration) {
    ROS_ASSERT(max_interval_duration >= ros::Duration(0,0));
    max_interval_duration_ = max_interval_duration;
  }

private:
  // Assumes that deque number <index> is non empty
  void dequeDeleteFront(int i)
  {
    if (i >= num_msgs_)
    {
      return;
    }

    Deque& deque = deques_[i];
    ROS_ASSERT(!deque.empty());
    deque.pop_front();
    if (deque.empty())
    {
      --num_non_empty_deques_;
    }
  }

  // Assumes that deque number <index> is non empty
  void dequeMoveFrontToPast(int i)
  {
    if (i >= num_msgs_)
    {
      return;
    }

    Deque& deque = deques_[i];
    Vector& vector = past_[i];
    ROS_ASSERT(!deque.empty());
    vector.push_back(deque.front());
    deque.pop_front();
    if (deque.empty())
    {
      --num_non_empty_deques_;
    }
  }

  void makeCandidate()
  {
    //printf("Creating candidate\n");
    // Create candidate tuple
    candidate_ = Tuple(num_msgs_); // Discards old one if any
    for (int i = 0; i < num_msgs_; ++i) {
      candidate_[i] = deques_[i].front();
    }
    // Delete all past messages, since we have found a better candidate
    for (int i = 0; i < num_msgs_; ++i) {
      past_[i].clear();
    }
    //printf("Candidate created\n");
  }


  // ASSUMES: num_messages <= past_[i].size()
  void recover(size_t num_messages, int i)
  {
    if (i >= num_msgs_)
    {
      return;
    }

    Vector& v= past_[i];
    Deque& q = deques_[i];
    ROS_ASSERT(num_messages <= v.size());
    while (num_messages > 0)
    {
      q.push_front(v.back());
      v.pop_back();
      num_messages--;
    }

    if (!q.empty())
    {
      ++num_non_empty_deques_;
    }
  }


  void recover(int i)
  {
    if (i >= num_msgs_)
    {
      return;
    }

    Vector& v= past_[i];
    Deque& q = deques_[i];
    while (!v.empty())
    {
      q.push_front(v.back());
      v.pop_back();
    }

    if (!q.empty())
    {
      ++num_non_empty_deques_;
    }
  }


  void recoverAndDelete(int i)
  {
    if (i >= num_msgs_)
    {
      return;
    }

    Vector& v= past_[i];
    Deque& q = deques_[i];
    while (!v.empty())
    {
      q.push_front(v.back());
      v.pop_back();
    }

    ROS_ASSERT(!q.empty());

    q.pop_front();
    if (!q.empty())
    {
      ++num_non_empty_deques_;
    }
  }

  // Assumes: all deques are non empty, i.e. num_non_empty_deques_ == RealTypeCount::value
  void publishCandidate()
  {
    //printf("Publishing candidate\n");
    // Publish
    parent_->signal(candidate_);
    // Delete this candidate
    candidate_ = Tuple(num_msgs_);
    pivot_ = NO_PIVOT;

    // Recover hidden messages, and delete the ones corresponding to the candidate
    num_non_empty_deques_ = 0; // We will recompute it from scratch
    for (int i = 0; i < num_msgs_; ++i) {
      recoverAndDelete(i);
    }
  }

  // Assumes: all deques are non empty, i.e. num_non_empty_deques_ == RealTypeCount::value
  // Returns: the oldest message on the deques
  void getCandidateStart(uint32_t &start_index, ros::Time &start_time)
  {
    return getCandidateBoundary(start_index, start_time, false);
  }

  // Assumes: all deques are non empty, i.e. num_non_empty_deques_ == RealTypeCount::value
  // Returns: the latest message among the heads of the deques, i.e. the minimum
  //          time to end an interval started at getCandidateStart_index()
  void getCandidateEnd(uint32_t &end_index, ros::Time &end_time)
  {
    return getCandidateBoundary(end_index, end_time, true);
  }

  // ASSUMES: all deques are non-empty
  // end = true: look for the latest head of deque
  //       false: look for the earliest head of deque
  void getCandidateBoundary(uint32_t &index, ros::Time &time, bool end)
  {
    namespace mt = ros::message_traits;

    Event& m0 = deques_[0].front();
    time = mt::TimeStamp<M>::value(*m0.getMessage());
    index = 0;
    for (int i = 1; i < num_msgs_; ++i) {
      Event& m1 = deques_[i].front();
      if ((mt::TimeStamp<M>::value(*m1.getMessage()) < time) ^ end)
      {
        time = mt::TimeStamp<M>::value(*m1.getMessage());
        index = i;
      }
    }
  }


  // ASSUMES: we have a pivot and candidate
  ros::Time getVirtualTime(int i)
  {
    namespace mt = ros::message_traits;

    if (i >= num_msgs_)
    {
      return ros::Time(0,0);  // Dummy return value
    }
    ROS_ASSERT(pivot_ != NO_PIVOT);

    Vector& v= past_[i];
    Deque& q = deques_[i];
    if (q.empty())
    {
      ROS_ASSERT(!v.empty());  // Because we have a candidate
      ros::Time last_msg_time = mt::TimeStamp<Message>::value(*(v.back()).getMessage());
      ros::Time msg_time_lower_bound = last_msg_time + inter_message_lower_bounds_[i];
      if (msg_time_lower_bound > pivot_time_)  // Take the max
      {
        return msg_time_lower_bound;
      }
      return pivot_time_;
    }
    ros::Time current_msg_time = mt::TimeStamp<Message>::value(*(q.front()).getMessage());
    return current_msg_time;
  }


  // ASSUMES: we have a pivot and candidate
  void getVirtualCandidateStart(uint32_t &start_index, ros::Time &start_time)
  {
    return getVirtualCandidateBoundary(start_index, start_time, false);
  }

  // ASSUMES: we have a pivot and candidate
  void getVirtualCandidateEnd(uint32_t &end_index, ros::Time &end_time)
  {
    return getVirtualCandidateBoundary(end_index, end_time, true);
  }

  // ASSUMES: we have a pivot and candidate
  // end = true: look for the latest head of deque
  //       false: look for the earliest head of deque
  void getVirtualCandidateBoundary(uint32_t &index, ros::Time &time, bool end)
  {
    namespace mt = ros::message_traits;

    std::vector<ros::Time> virtual_times(num_msgs_);
    for (int i = 0; i < num_msgs_; ++i) {
      virtual_times[i] = getVirtualTime(i);
    }
 
    time = virtual_times[0];
    index = 0;
    for (int i = 0; i < num_msgs_; i++)
    {
      if ((virtual_times[i] < time) ^ end)
      {
        time = virtual_times[i];
        index = i;
      }
    }
  }


  // assumes data_mutex_ is already locked
  void process()
  {
    // While no deque is empty
    while (num_non_empty_deques_ == num_msgs_)
    {
      // Find the start and end of the current interval
      //printf("Entering while loop in this state [\n");
      //show_internal_state();
      //printf("]\n");
      ros::Time end_time, start_time;
      uint32_t end_index, start_index;
      getCandidateEnd(end_index, end_time);
      getCandidateStart(start_index, start_time);
      //ROS_INFO("processing...: start_idx: %d, end_idx: %d", start_index, end_index);
      for (uint32_t i = 0; i < num_msgs_; i++)
      {
        if (i != end_index)
        {
          // No dropped message could have been better to use than the ones we have,
          // so it becomes ok to use this topic as pivot in the future
          has_dropped_messages_[i] = false;
        }
      }
      if (pivot_ == NO_PIVOT)
      {
        //ROS_INFO("We do not have a candidate, pivot: %d", pivot_);
        // We do not have a candidate
        // INVARIANT: the past_ vectors are empty
        // INVARIANT: (candidate_ has no filled members)
        if (end_time - start_time > max_interval_duration_)
        {
          // This interval is too big to be a valid candidate, move to the next
          dequeDeleteFront(start_index);
          continue;
        }
        if (has_dropped_messages_[end_index])
        {
          // The topic that would become pivot has dropped messages, so it is not a good pivot
          dequeDeleteFront(start_index);
          continue;
        }
        // This is a valid candidate, and we don't have any, so take it
        makeCandidate();
        candidate_start_ = start_time;
        candidate_end_ = end_time;
        pivot_ = end_index;
        pivot_time_ = end_time;
        //ROS_INFO("dequeMoveFrontToPast, start_idx: %d, end_idx: %d", start_index, end_index);
        dequeMoveFrontToPast(start_index);
        //ROS_INFO("..done dequeMoveFrontToPast");
      }
      else
      {
        //ROS_INFO("We already have a candidate, pivto: %d", pivot_);
        // We already have a candidate
        // Is this one better than the current candidate?
        // INVARIANT: has_dropped_messages_ is all false
        if ((end_time - candidate_end_) * (1 + age_penalty_) >= (start_time - candidate_start_))
        {
          // This is not a better candidate, move to the next
          dequeMoveFrontToPast(start_index);
        }
        else
        {
          // This is a better candidate
          makeCandidate();
          candidate_start_ = start_time;
          candidate_end_ = end_time;
          //ROS_INFO("dequeMoveFrontToPast, start_idx: %d, end_idx: %d", start_index, end_index);
          dequeMoveFrontToPast(start_index);
          //ROS_INFO("..done dequeMoveFrontToPast");
          // Keep the same pivot (and pivot time)
        }
      }
      // INVARIANT: we have a candidate and pivot
      ROS_ASSERT(pivot_ != NO_PIVOT);
      //printf("start_index == %d, pivot_ == %d\n", start_index, pivot_);
      if (start_index == pivot_)  // TODO: replace with start_time == pivot_time_
      {
        // We have exhausted all possible candidates for this pivot, we now can output the best one
        publishCandidate();
      }
      else if ((end_time - candidate_end_) * (1 + age_penalty_) >= (pivot_time_ - candidate_start_))
      {
        // We have not exhausted all candidates, but this candidate is already provably optimal
        // Indeed, any future candidate must contain the interval [pivot_time_ end_time], which
        // is already too big.
        // Note: this case is subsumed by the next, but it may save some unnecessary work and
        //       it makes things (a little) easier to understand
        publishCandidate();
      }
      else if (num_non_empty_deques_ < num_msgs_)
      {
        uint32_t num_non_empty_deques_before_virtual_search = num_non_empty_deques_;

        // Before giving up, use the rate bounds, if provided, to further try to prove optimality
        std::vector<int> num_virtual_moves(num_msgs_,0);
        //ROS_INFO("while (1)... virtual move");
        while (1)
        {
          ros::Time end_time, start_time;
          uint32_t end_index, start_index;
          getVirtualCandidateEnd(end_index, end_time);
          getVirtualCandidateStart(start_index, start_time);
          if ((end_time - candidate_end_) * (1 + age_penalty_) >= (pivot_time_ - candidate_start_))
          {
            // We have proved optimality
            // As above, any future candidate must contain the interval [pivot_time_ end_time], which
            // is already too big.
            publishCandidate();  // This cleans up the virtual moves as a byproduct
            break;  // From the while(1) loop only
          }
          if ((end_time - candidate_end_) * (1 + age_penalty_) < (start_time - candidate_start_))
          {
            // We cannot prove optimality
            // Indeed, we have a virtual (i.e. optimistic) candidate that is better than the current
            // candidate
            // Cleanup the virtual search:
            num_non_empty_deques_ = 0; // We will recompute it from scratch
            for (int i = 0; i < num_msgs_; ++i) {
              recover(num_virtual_moves[i], i);
            }
            (void)num_non_empty_deques_before_virtual_search; // unused variable warning stopper
            ROS_ASSERT(num_non_empty_deques_before_virtual_search == num_non_empty_deques_);
            break;
          }
          // Note: we cannot reach this point with start_index == pivot_ since in that case we would
          //       have start_time == pivot_time, in which case the two tests above are the negation
          //       of each other, so that one must be true. Therefore the while loop always terminates.
          ROS_ASSERT(start_index != pivot_);
          ROS_ASSERT(start_time < pivot_time_);
          dequeMoveFrontToPast(start_index);
          num_virtual_moves[start_index]++;
        } // while(1)
        //ROS_INFO("End while (1)");
      }
    } // while(num_non_empty_deques_ == num_msgs_)
    //ROS_INFO("... done process...");
  }

  Sync* parent_;
  uint32_t queue_size_;
  uint32_t num_msgs_;

  uint32_t NO_PIVOT;  // Special value for the pivot indicating that no pivot has been selected

  DequeTuple deques_;
  uint32_t num_non_empty_deques_;
  VectorTuple past_;
  Tuple candidate_;  // NULL if there is no candidate, in which case there is no pivot.
  ros::Time candidate_start_;
  ros::Time candidate_end_;
  ros::Time pivot_time_;
  uint32_t pivot_;  // Equal to NO_PIVOT if there is no candidate
  boost::mutex data_mutex_;  // Protects all of the above

  ros::Duration max_interval_duration_; // TODO: initialize with a parameter
  double age_penalty_;

  std::vector<bool> has_dropped_messages_;
  std::vector<ros::Duration> inter_message_lower_bounds_;
  std::vector<bool> warned_about_incorrect_bound_;
};

} // namespace sync
} // namespace message_filters

#endif // MESSAGE_FILTERS_SYNC_APPROXIMATE_TIME_H

