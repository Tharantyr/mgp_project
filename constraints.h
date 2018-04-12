#ifndef CONSTRAINTS_HEADER_FILE
#define CONSTRAINTS_HEADER_FILE

using namespace Eigen;
using namespace std;

typedef enum ConstraintType{DISTANCE, EXTENSION, COMPRESSION, COLLISION, BARRIER} ConstraintType;   //seems redundant, but you can expand it
typedef enum ConstraintEqualityType{EQUALITY, INEQUALITY} ConstraintEqualityType;

//there is such constraints per two variables that are equal. That is, for every attached vertex there are three such constraints for (x,y,z);
class Constraint{
public:
  
  VectorXi globalIndices;         //|V_c| list of participating indices (out of global indices of the scene)
  double currValue;               //The current value of the constraint
  RowVectorXd currGradient;       //Current gradient of the constraint
  MatrixXd invMassMatrix;         //M^{-1} matrix size |V_c| X |V_c| only over the participating  vertices
  double refValue;                //Reference values to use in the constraint, when needed
  VectorXd refVector;             //Reference vector when needed
  double CRCoeff;
  ConstraintType constraintType;  //The type of the constraint, and will affect the value and the gradient. This SHOULD NOT change after initialization!
  ConstraintEqualityType constraintEqualityType;  //whether the constraint is an equality or an inequality
  
  Constraint(const ConstraintType _constraintType, const ConstraintEqualityType _constraintEqualityType, const VectorXi& _globalIndices, const VectorXd& invMasses, const VectorXd& _refVector, const double _refValue, const double _CRCoeff):constraintType(_constraintType), constraintEqualityType(_constraintEqualityType), refVector(_refVector), refValue(_refValue), CRCoeff(_CRCoeff){
    currValue=0.0;
    globalIndices=_globalIndices;
    currGradient=VectorXd::Zero(globalIndices.size());
    invMassMatrix=invMasses.asDiagonal();
  }
  
  ~Constraint(){}
  
  
  //updating the value and the gradient vector with given values
  void updateValueGradient(const VectorXd& currVars){
    switch (constraintType){
        
      case DISTANCE: {
        /***************************
         TODO
         ***************************/

		  VectorXd v1 = VectorXd(3);
		  v1(0) = currVars(0);
		  v1(1) = currVars(1);
		  v1(2) = currVars(2);

		  VectorXd v2 = VectorXd(3);
		  v2(0) = currVars(3);
		  v2(1) = currVars(4);
		  v2(2) = currVars(5);

		  VectorXd v1minv2 = v1 - v2;

		  double length = sqrt(v1minv2.dot(v1minv2));

		  currGradient(0) = (v1(0) - v2(0)) / length;
		  currGradient(1) = (v1(1) - v2(1)) / length;
		  currGradient(2) = (v1(2) - v2(2)) / length;
		  currGradient(3) = -(v1(0) - v2(0)) / length;
		  currGradient(4) = -(v1(1) - v2(1)) / length;
		  currGradient(5) = -(v1(2) - v2(2)) / length;

		  currValue = length - refValue;

        break;
      }
        
	  case EXTENSION: {

		  VectorXd v1 = VectorXd(3);
		  v1(0) = currVars(0);
		  v1(1) = currVars(1);
		  v1(2) = currVars(2);

		  VectorXd v2 = VectorXd(3);
		  v2(0) = currVars(3);
		  v2(1) = currVars(4);
		  v2(2) = currVars(5);

		  VectorXd v1minv2 = v1 - v2;

		  double length = sqrt(v1minv2.dot(v1minv2));

		  currGradient(0) = (v1(0) - v2(0)) / length;
		  currGradient(1) = (v1(1) - v2(1)) / length;
		  currGradient(2) = (v1(2) - v2(2)) / length;
		  currGradient(3) = -(v1(0) - v2(0)) / length;
		  currGradient(4) = -(v1(1) - v2(1)) / length;
		  currGradient(5) = -(v1(2) - v2(2)) / length;

		  currValue = length - refValue;

		  break;
	  }

	  case COMPRESSION: {

		  VectorXd v1 = VectorXd(3);
		  v1(0) = currVars(0);
		  v1(1) = currVars(1);
		  v1(2) = currVars(2);

		  VectorXd v2 = VectorXd(3);
		  v2(0) = currVars(3);
		  v2(1) = currVars(4);
		  v2(2) = currVars(5);

		  VectorXd v1minv2 = v1 - v2;

		  double length = sqrt(v1minv2.dot(v1minv2));

		  currGradient(0) = -(v1(0) - v2(0)) / length;
		  currGradient(1) = -(v1(1) - v2(1)) / length;
		  currGradient(2) = -(v1(2) - v2(2)) / length;
		  currGradient(3) = (v1(0) - v2(0)) / length;
		  currGradient(4) = (v1(1) - v2(1)) / length;
		  currGradient(5) = (v1(2) - v2(2)) / length;

		  currValue = refValue - length;

		  break;
	  }

      case COLLISION:{
        /***************************
         TODO
         ***************************/

		  currValue = refVector.dot(currVars);
		  currGradient = refVector;

        break;
      }
        
      case BARRIER:{
        /***************************
         TODO
         ***************************/

		  currValue = currVars(0);
		  currGradient(0) = 1;

		  //if (globalIndices(0) == 1)
			//printf("%f\n", currValue);

        break;
      }
    }
  }
  
  
  //computes the impulse needed for all particles to resolve the velocity constraint
  //returns true if constraint was already good
  bool resolveVelocityConstraint(const VectorXd& currPositions, const VectorXd& currVelocities, VectorXd& newImpulses, double tolerance){
    
    updateValueGradient(currPositions);
    
    if ((constraintEqualityType==INEQUALITY)&&((currValue>-tolerance)||((currGradient*currVelocities)(0,0)>-tolerance))){
      //constraint is valid, or velocity is already solving it, so nothing happens
      newImpulses=VectorXd::Zero(globalIndices.size());
      return true;
    }
    
    if ((constraintEqualityType==EQUALITY)&&(abs((currGradient*currVelocities)(0,0))<tolerance)){
      newImpulses=VectorXd::Zero(globalIndices.size());
      return true;
    }
    
    /***************************
     TODO
     ***************************/

	VectorXd lambda = (-currGradient * currVelocities) / (currGradient * invMassMatrix * currGradient.transpose());
	VectorXd impulses = invMassMatrix * currGradient.transpose() * lambda;

	if (constraintEqualityType==INEQUALITY)
		impulses *= (1 + CRCoeff);

	newImpulses = impulses;

    return false;
  }
  
  //projects the position unto the constraint
  //returns true if constraint was already good
  bool resolvePositionConstraint(const VectorXd& currPositions, const VectorXd& currVelocities, VectorXd& newPosDiffs, double tolerance){
    
    updateValueGradient(currPositions);
    //cout<<"C(currPositions): "<<currValue<<endl;
    //cout<<"currPositions: "<<currPositions<<endl;
    if ((constraintEqualityType==INEQUALITY)&&(currValue>-tolerance)){
      //constraint is valid
      newPosDiffs=VectorXd::Zero(globalIndices.size());
      return true;
    }
    
    if ((constraintEqualityType==EQUALITY)&&(abs(currValue)<tolerance)){
      newPosDiffs=VectorXd::Zero(globalIndices.size());
      return true;
    }
   
    /***************************
     TODO
     ***************************/
   
	double lambda = (-currValue) / (currGradient * invMassMatrix * currGradient.transpose());
	newPosDiffs = lambda * invMassMatrix * currGradient;

    return false;
  }
};



#endif /* constraints_h */
