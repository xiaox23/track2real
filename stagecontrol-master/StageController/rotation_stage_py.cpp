#include <boost/python.hpp>
#include "RotationStage.h"

using namespace boost::python;

BOOST_PYTHON_MODULE(rotation_stage)
{
	class_<RotationStage>("RotationStage", init<const char*, int>())
		.def("seek_and_set_zero", &RotationStage::seek_and_set_zero)
		.def("rel_move", &RotationStage::rel_move)
        .def("abs_move", &RotationStage::abs_move)
		.def("stop", &RotationStage::stop)
        .def("get_position", &RotationStage::get_position)
		.def("get_velocity", &RotationStage::get_velocity)
        .def("is_moving", &RotationStage::is_moving)
		.def("wait_for_motion_finished", &RotationStage::wait_for_motion_finished)
        ;
}

