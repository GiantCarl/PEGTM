// Gmsh project created on Sat Sep 21 08:45:24 2024
SetFactory("OpenCASCADE");
//+
w = 1.0;   //width of specimen
//+
h = 1.0;  //High of specimen
//+
w1 = 0.01; //mesh refine region width
//+
dx = 0.04 ;//mesh size
//+
dx1=0.002;//efine mesh size
//+
Point(1) = {-0.5,   -0.5,   0.0,    dx};    //left-bottom point
//+
Point(2) = {-0.5+w, -0.5,   0.0,    dx};    //right-bottom point
//+
Point(3) = {-0.5+w, -0.5+h, 0.0,    dx};    //right-top point
//+
Point(4) = {-0.5,   -0.5+h, 0.0,    dx};    //left-top point
//+
Point(5) = {-0.5+0.25,  -0.5,   0.0,    dx};
//+
Point(6) = {-0.5+0.75,  -0.5,   0.0,    dx};
//+
Point(7) = {-0.5+0.25,  0.5,   0.0,    dx};
//+
Point(8) = {-0.5+0.75,  0.5,   0.0,    dx};
//+
Point(9) = {0.0,  -0.5,   0.0,    dx};
//+
Point(10) = {0.0,  0.5,   0.0,    dx};
//+
Line(1) = {1, 5};
//+
Line(2) = {6, 2};
//+
Line(3) = {2, 3};
//+
Line(4) = {3, 8};
//+
Line(5) = {7, 4};
//+
Line(6) = {4, 1};
//+
Circle(7) = {5, 9, 6};
//+
Circle(8) = {8, 10, 7};
//+
Curve Loop(1) = {6, 1, 7, 2, 3, 4, 8, 5};
//+
Plane Surface(1) = {1};

//+
Physical Point("bottom_left", 9) = {1};
//+
Physical Point("bottom_right", 10) = {2};
//+
Physical Point("top_right", 11) = {3};
//+
Physical Point("top_left", 12) = {4};
//+
Physical Curve("left_edge", 13) = {6};
//+
Physical Curve("bottom_edge", 14) = {1, 7, 2};
//+
Physical Curve("right_edge", 15) = {3};
//+
Physical Curve("top_edge", 16) = {4, 8, 5};
//+
Physical Surface("block", 17) = {1};

//+
Field[1] = Box;
//+
Field[1].Thickness = 0.1;
//+
Field[1].VIn = 0.002;
//+
Field[1].VOut = 0.04;
//+
Field[1].XMax = 0.05;
//+
Field[1].XMin = -0.05;
//+
Field[1].YMax = 0.5;
//+
Field[1].YMin = -0.5;
//+
Background Field = 1;

