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

#ifndef MESSAGE_FILTERS_SYNCHRONIZER_VEC_H
#define MESSAGE_FILTERS_SYNCHRONIZER_VEC_H

#include <boost/tuple/tuple.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
#include <boost/thread/mutex.hpp>

#include <boost/bind.hpp>
#include <boost/noncopyable.hpp>

#include <message_filters/connection.h>
#include <ros/assert.h>
#include <ros/message_traits.h>
#include <ros/message_event.h>

#include <deque>
#include <vector>
#include <string>

namespace message_filters
{

template<class Policy>
class SynchronizerVec : public boost::noncopyable, public Policy
{
public:
  typedef typename Policy::Message Message;
  typedef typename Policy::Event Event;
  typedef boost::function<void(const std::vector<typename Event::ConstMessagePtr>& ) > Callback;

  template<class F>
  SynchronizerVec(std::vector<F>& fs)
    : num_msgs_(fs.size()), input_connections_(fs.size()), callback_registered_(false)
  {
    connectInput(fs);
    init();
  }

  SynchronizerVec()
    : callback_registered_(false)
  {
    init();
  }

  template<class F>
  SynchronizerVec(const Policy& policy, std::vector<F>& fs)
  : Policy(policy), num_msgs_(fs.size()), input_connections_(fs.size()), callback_registered_(false)
  {
    connectInput(fs);
    init();
  }

  SynchronizerVec(const Policy& policy)
  : Policy(policy), callback_registered_(false)
  {
    init();
  }

  ~SynchronizerVec()
  {
    disconnectAll();
  }

  void init()
  {
    Policy::initParent(this);
  }

  template<class F>
  void connectInput(std::vector<F>& fs)
  {
    disconnectAll();

    for (int i = 0; i < fs.size(); ++i) {
      input_connections_[i] = fs[i].registerCallback(boost::function<void(const Event&)>(boost::bind(&SynchronizerVec::cb, this, _1, i)));
    }
    //ROS_INFO("registered %zu input connections", fs.size());
  }

  template<class C>
  void registerCallback(C& callback)
  {
    //ROS_INFO("Registering synced callback");
    callback_ = callback;
    callback_registered_ = true;
  }

  template<class C>
  void registerCallback(const C& callback)
  {
    callback_ = callback;
    callback_registered_ = true;
  }

// TODO
//  template<class C, typename T>
//  void registerCallback(const C& callback, T* t)
//  {
//    return signal_.addCallback(callback, t);
//  }

//  template<class C, typename T>
//  void registerCallback(C& callback, T* t)
//  {
//    return signal_.addCallback(callback, t);
//  }

  void setName(const std::string& name) { name_ = name; }
  const std::string& getName() { return name_; }


  void signal(const std::vector<Event>& es)
  {
    //ROS_INFO("synchronizer signal called!");
    if(!callback_registered_){
      ROS_WARN("synchronizer signal called but no callback registered! aborting.");
      return;
    }

    std::vector<typename Event::ConstMessagePtr> msgs(es.size());
    for (int i = 0; i < es.size(); ++i) {
      msgs[i] = es[i].getConstMessage();
    }
    //ROS_INFO("synchronizer calling callback");
    callback_(msgs);
  }

  int get_num_msgs() { return num_msgs_;}
  Policy* getPolicy() { return static_cast<Policy*>(this); }

  using Policy::add;

  void add(const boost::shared_ptr<Message const>& msg, int i)
  {
    //ROS_INFO("Synchronizer add msg %d", i);
    this->add(Event(msg), i);
  }

private:

  void disconnectAll()
  {
    for (int i = 0; i < num_msgs_; ++i)
    {
      input_connections_[i].disconnect();
    }
  }

  void cb(const Event& evt, int i)
  {
    //ROS_INFO("Synchronizer input Calback event %d", i);
    this->add(evt, i);
  }

  uint32_t queue_size_;
  uint32_t num_msgs_;

  bool callback_registered_;
  Callback callback_;

  std::vector<Connection> input_connections_;

  std::string name_;
};

} // namespace message_filters

#endif // MESSAGE_FILTERS_SYNCHRONIZER_H
