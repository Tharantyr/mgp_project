#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/unproject.h>
#include <imgui/imgui.h>
#include <iostream>
#include "scene.h"
#include <igl/copyleft/tetgen/tetrahedralize.h>

using namespace Eigen;
using namespace std;

MatrixXd V;
MatrixXi F;
MatrixXd P1, P2;   //for marking constraints
igl::opengl::glfw::Viewer mgpViewer;
int distConstV1, distConstV2;
Eigen::MatrixXi EConst;    //the user distance constraints in (V1,V2) format each row

MatrixXd platTriV;
MatrixXi platTriF;

float currTime = 0;
float timeStep = 0.02;
float CRCoeff= 1.0;
double tolerance=10e-6;
int maxIterations=100;
bool vertexChosen=false;
int currVertex=-1;
int userconstraint = 0;
Scene scene;



void createPlatform(MatrixXd& platV, MatrixXi& platF, RowVector3d& platCOM, RowVector4d& platOrientation)
{
  double platWidth=100.0;
  platCOM<<0.0,0.0,-0.0;
  platV.resize(8,3);
  platF.resize(12,3);
  platV<<-platWidth,0.0,-platWidth,
  -platWidth,0.0,platWidth,
  platWidth,0.0,platWidth,
  platWidth,0.0, -platWidth,
  -platWidth,-platWidth/2.0,-platWidth,
  -platWidth,-platWidth/2.0,platWidth,
  platWidth,-platWidth/2.0,platWidth,
  platWidth,-platWidth/2.0, -platWidth;
  platF<<0,1,2,
  2,3,0,
  6,5,4,
  4,7,6,
  1,0,5,
  0,4,5,
  2,1,6,
  1,5,6,
  3,2,7,
  2,6,7,
  0,3,4,
  3,7,4;
  
  platOrientation<<1.0,0.0,0.0,0.0;
}


void update_mesh(igl::opengl::glfw::Viewer &viewer)
{
  viewer.data().clear();
  MatrixXi fullF(platTriF.rows()+F.rows(),3);
  fullF<<platTriF, F.array()+platTriV.rows();
  MatrixXd fullV(platTriV.rows()+V.rows(),3);
  fullV<<platTriV, V;
  viewer.data().set_mesh(fullV,fullF);
 
  Eigen::MatrixXd constV1(EConst.rows(),3), constV2(EConst.rows(),3);
  for (int i=0;i<EConst.rows();i++){
    constV1.row(i)=V.row(EConst(i,0));
    constV2.row(i)=V.row(EConst(i,1));
  }
  
  RowVector3d constColor; constColor<<0.2,1.0,0.1;
  viewer.data().set_face_based(true);
  viewer.data().add_edges(constV1, constV2, constColor);
}

bool key_down(igl::opengl::glfw::Viewer &viewer, unsigned char key, int modifier)
{
	double x = viewer.current_mouse_x;
	double y = viewer.core.viewport(3) - viewer.current_mouse_y;

  if (key == ' ')
  {
    viewer.core.is_animating = !viewer.core.is_animating;
    return true;
  }
  
  if (key == 'S')
  {
    if (!viewer.core.is_animating){
      scene.updateScene(timeStep, CRCoeff, tolerance, maxIterations,V);
      currTime+=timeStep;
      update_mesh(viewer);
      std::cout <<"currTime: "<<currTime<<std::endl;
      return true;
    }
  }

  //smash all meshes to a sphere
  if (key == 'P') {
	  //calculate centre of sphere
	  for (int mesh = 0; mesh < scene.meshes.size(); mesh++) {
		  double x, y, z;
		  for (int i = 0; i < scene.meshes[mesh].currPositions.size() / 3; i++) {
			  x += scene.meshes[mesh].currPositions(i * 3);
			  y += scene.meshes[mesh].currPositions(i * 3 + 1);
			  z += scene.meshes[mesh].currPositions(i * 3 + 2);
		  }
		  x /= (scene.meshes[mesh].currPositions.size() / 3);
		  y /= (scene.meshes[mesh].currPositions.size() / 3);
		  z /= (scene.meshes[mesh].currPositions.size() / 3);
		  Vector3d centre(x, y, z);

		  for (int i = 0; i < scene.meshes[mesh].currPositions.size() / 3; i++) {
			  //get normal vector from centre to point
			  Vector3d vec = Vector3d(scene.meshes[mesh].currPositions(i * 3), scene.meshes[mesh].currPositions(i * 3 + 1), scene.meshes[mesh].currPositions(i * 3 + 2)) - centre;
			  vec.normalize();

			  //calculate new pas and insert in curr pos
			  Vector3d newPos = centre + vec;
			  scene.meshes[mesh].currPositions(i * 3) = newPos(0);
			  scene.meshes[mesh].currPositions(i * 3 + 1) = newPos(1);
			  scene.meshes[mesh].currPositions(i * 3 + 2) = newPos(2);
		  }
	  }
  }
  // Toggle gravity (default enabled)
  if (key == 'G') {
	  for (int i = 0; i < scene.meshes.size(); i++)
		  scene.meshes[i].enableGravity = !scene.meshes[i].enableGravity;
	  if (scene.meshes[0].enableGravity)
		  cout << "Gravity is enabled" << endl;
	  else
		  cout << "Gravity is disabled" << endl;
  }

  if (key == 'C') {
	  if (userconstraint == 0)
	  {
		  userconstraint = 1;
		  cout << "Set to extension constraint" << endl;
		  return true;
	  }
	  if (userconstraint == 1)
	  {
		  userconstraint = 2;
		  cout << "Set to compression constraint" << endl;
		  return true;
	  }
	  if (userconstraint == 2)
	  {
		  userconstraint = 0;
		  cout << "Set to distance constraint" << endl;
		  return true;
	  }
  }

  if (key == 'B')
  {
	  scene.destroyConstraints = !scene.destroyConstraints;
	  if (scene.destroyConstraints)
		  cout << "Constraints destroyed" << endl;
	  else
		  cout << "Constraints not destroyed" << endl;
  }

  return false;
}


bool pre_draw(igl::opengl::glfw::Viewer &viewer)
{
  using namespace Eigen;
  using namespace std;
  
  if (viewer.core.is_animating){
    scene.updateScene(timeStep, CRCoeff, tolerance, maxIterations, V);
    update_mesh(viewer);
    currTime+=timeStep;
    //cout <<"currTime: "<<currTime<<endl;
  }
  
  return false;
}

class CustomMenu : public igl::opengl::glfw::imgui::ImGuiMenu
{
  
  virtual void draw_viewer_menu() override
  {
    // Draw parent menu
    ImGuiMenu::draw_viewer_menu();
    
    // Add new group
    if (ImGui::CollapsingHeader("Algorithm Options", ImGuiTreeNodeFlags_DefaultOpen))
    {
      ImGui::InputFloat("CR Coeff",&CRCoeff,0,0,3);
      
      
      if (ImGui::InputFloat("Time Step", &timeStep)) {
        mgpViewer.core.animation_max_fps = (((int)1.0/timeStep));
        scene.initScene(timeStep, 0.02, 0.02, V);
      }
    }
  }
};

bool mouse_up(igl::opengl::glfw::Viewer& viewer, int button, int modifiers)
{
  if (!vertexChosen)
    return false;
  double x = viewer.current_mouse_x;
  double y = viewer.core.viewport(3) - viewer.current_mouse_y;
  Vector3f newPos=igl::unproject(Vector3f(x, y, viewer.down_mouse_z), (viewer.core.view * viewer.core.model).eval(), viewer.core.proj, viewer.core.viewport);
  
  if ((igl::opengl::glfw::Viewer::MouseButton)button==igl::opengl::glfw::Viewer::MouseButton::Left){
    scene.createUserImpulse(currVertex, newPos);
    update_mesh(viewer);
    vertexChosen=false;
    return true;
  }
  
  if ((igl::opengl::glfw::Viewer::MouseButton)button==igl::opengl::glfw::Viewer::MouseButton::Right){
    //creating new constraint
    int fid;
    Eigen::Vector3f bc;
    if(igl::unproject_onto_mesh(Eigen::Vector2f(x,y), viewer.core.view * viewer.core.model,
                                viewer.core.proj, viewer.core.viewport, V, F, fid, bc));
    
    Eigen::MatrixXf::Index maxCol;
    bc.maxCoeff(&maxCol);
    int otherVertex=F(fid, maxCol);

	if (userconstraint == 0)
		scene.addUserConstraint(currVertex, otherVertex, EConst);
	if (userconstraint == 1)
		scene.addExtensionConstraint(currVertex, otherVertex, EConst);
	if (userconstraint == 2)
		scene.addCompressionConstraint(currVertex, otherVertex, EConst);

    update_mesh(viewer);
    vertexChosen=false;
    return true;
  }
  
  return false;
}

bool mouse_down(igl::opengl::glfw::Viewer& viewer, int button, int modifiers)
{
  int fid;
  Eigen::Vector3f bc;
  // Cast a ray in the view direction starting from the mouse position
  double x = viewer.current_mouse_x;
  double y = viewer.core.viewport(3) - viewer.current_mouse_y;
  
  if(igl::unproject_onto_mesh(Eigen::Vector2f(x,y), viewer.core.view * viewer.core.model,
                              viewer.core.proj, viewer.core.viewport, V, F, fid, bc))
  {
    //giving impulse to the chosen vertex
    Eigen::MatrixXf::Index maxCol;
    bc.maxCoeff(&maxCol);
    currVertex=F(fid, maxCol);
    vertexChosen=true;
    mgpViewer.core.is_animating = false;

    if ((igl::opengl::glfw::Viewer::MouseButton)button==igl::opengl::glfw::Viewer::MouseButton::Left){
      //TODO: give impulses --> Handled in mouse_up and scene.h
    }
    
    if ((igl::opengl::glfw::Viewer::MouseButton)button==igl::opengl::glfw::Viewer::MouseButton::Right){
    }
    return true;
  }
  return false;
};


int main(int argc, char *argv[])
{
  using namespace Eigen;
  using namespace std;
  
  
  // Load scene
  if (argc<3){
    cout<<"Run program as follows: infomgp_practical2 <folder_name> <scene_file> <constraint_file>"<<endl;
    return 0;
  }
  cout<<"data folder: "<<std::string(argv[1])<<endl;
  cout<<"scene file: "<<std::string(argv[2])<<endl;
  cout<<"constraints file: "<<std::string(argv[3])<<endl;
 
  
  //create platform

  RowVector3d platCOM;
  RowVector4d platOrientation;
  createPlatform(platTriV, platTriF, platCOM, platOrientation);
  //igl::copyleft::tetgen::tetrahedralize(platTriV,platTriF,"pq1.2", platV,platT,platF);
  //scene.addMesh(platV, platT, 0, 0, 100000.0, true, platCOM, platOrientation);
  
  scene.loadScene(std::string(argv[1]),std::string(argv[2]),std::string(argv[3]), F, EConst);
  scene.initScene(timeStep, 0.02, 0.02, V);
  scene.setPlatformBarriers(platTriV, CRCoeff);
  
  //cout<<"F: "<<F<<endl;
  //cout<<"platV: "<<platV<<endl;
  
  // Viewer Settings
  
  mgpViewer.callback_pre_draw = &pre_draw;
  mgpViewer.callback_key_down = &key_down;
  mgpViewer.callback_mouse_up = &mouse_up;
  mgpViewer.callback_mouse_down = &mouse_down;
  mgpViewer.core.is_animating = false;
  mgpViewer.core.animation_max_fps = 50.;
  
  CustomMenu menu;
  mgpViewer.plugins.push_back(&menu);
  
  update_mesh(mgpViewer);
  
  cout<<"Press [space] to toggle continuous simulation." << endl;
  cout<<"Press 'S' to advance time step-by-step."<<endl;
  cout<<"Left mouse button - Pick, drag, and drop vertices by mouse to give them impulses in direction of the drag."<<endl;
  cout<<"Right mouse button - Pick, drag, and pick another vertex to create an attachment constraint between them."<<endl;
  mgpViewer.launch();
}
