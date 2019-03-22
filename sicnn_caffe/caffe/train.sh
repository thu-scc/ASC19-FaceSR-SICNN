while true;do 
	./build/tools/caffe train     --solver=./face_example/face_solver.prototxt -gpu=0,1;sleep 5;done