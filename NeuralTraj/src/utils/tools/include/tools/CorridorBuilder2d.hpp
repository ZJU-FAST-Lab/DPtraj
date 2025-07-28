#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <random>
#include <decomp_basis/data_type.h>
#include <Eigen/Eigen>
#include "gridmap.hpp"

namespace plan_utils
{

  void corridorBuilder2d(Eigen::Vector2d origin, float radius, float max_x, float max_y,
                        vec_Vec2f &data, std::vector<Eigen::Vector2d> &add_data, 
                        Eigen::MatrixXd &hPoly) {

    //max_x max_y may be the local map?
    //why add_data?
    //radius = inf?
    //origin is the root point of the poly
    float interior_x = 0.0;
    float interior_y = 0.0;
    float safe_radius = radius;
    std::vector<cv::Point2f> flipData;

    cv::Point2f point;
    vec_Vec2f new_data;


    for (size_t i=0; i<data.size(); i++) {
        float dx = data[i](0) - origin(0);
        float dy = data[i](1) - origin(1);
        float norm2 = std::sqrt(dx*dx + dy*dy);
        if ( abs(dx) > max_x || abs(dy) > max_y) {
        continue;
        }
        if (norm2 < safe_radius) safe_radius = norm2; //safe radius is the nearest distance between the root and obs
        if (norm2 == 0) continue;
        
        point.x = dx + 2*(radius-norm2)*dx/norm2;
        point.y = dy + 2*(radius-norm2)*dy/norm2; //radius is an  enough large value
        new_data.push_back(data[i]);
        flipData.push_back(point);
    }

    for (size_t i=0; i<add_data.size(); i++) {
        float dx = add_data[i](0) - origin(0);
        float dy = add_data[i](1) - origin(1);
        float norm2 = std::sqrt(dx*dx + dy*dy);
        if (norm2 < safe_radius) safe_radius = norm2;
        if (norm2 == 0) continue;
        point.x = dx + 2*(radius-norm2)*dx/norm2;
        point.y = dy + 2*(radius-norm2)*dy/norm2;
        new_data.push_back(add_data[i]);
        flipData.push_back(point);
    }

    std::vector<int> vertexIndice;
    cv::convexHull(flipData,vertexIndice,false,false);
    //obtain the poly containing flipData
    
    bool isOriginAVertex = false;
    int OriginIndex = -1;
    std::vector<cv::Point2f> vertexData;
    for (size_t i=0; i<vertexIndice.size(); i++) {
        unsigned int v = vertexIndice[i];
        if (v == new_data.size()) {
            isOriginAVertex = true;
            OriginIndex = i;
            vertexData.push_back(cv::Point2f(origin(0), origin(1)));
        }else {
            vertexData.push_back(cv::Point2f(new_data[v](0), new_data[v](1)));
        }
    }

    if (isOriginAVertex) {
        int last_index = (OriginIndex - 1)%vertexIndice.size();
        int next_index = (OriginIndex + 1)%vertexIndice.size();
        float dx = (new_data[vertexIndice[last_index]](0) + origin(0) + new_data[vertexIndice[next_index]](0))/3 - origin(0);
        float dy = (new_data[vertexIndice[last_index]](1) + origin(1) + new_data[vertexIndice[next_index]](1))/3 - origin(1);
        float d = std::sqrt(dx*dx + dy*dy);
        interior_x = 0.99*safe_radius*dx/d + origin(0);
        interior_y = 0.99*safe_radius*dy/d + origin(1);
    }else {
        interior_x = origin(0);
        interior_y = origin(1);
    }

    std::vector<int> vIndex2;
    cv::convexHull(vertexData,vIndex2,false,false); // counterclockwise right-hand

    

    std::vector<Eigen::Vector3f> constraints; // (a,b,c) a x + b y <= c
    for (size_t j=0; j<vIndex2.size(); j++) {
        int jplus1 = (j+1)%vIndex2.size();
        cv::Point2f rayV = vertexData[vIndex2[jplus1]] - vertexData[vIndex2[j]];
        Eigen::Vector2f normalJ(rayV.y, -rayV.x);  // point to outside
        normalJ.normalize();
        int indexJ = vIndex2[j];
        while (indexJ != vIndex2[jplus1]) {
            float c = (vertexData[indexJ].x-interior_x) * normalJ(0) + (vertexData[indexJ].y-interior_y) * normalJ(1);
            constraints.push_back(Eigen::Vector3f(normalJ(0), normalJ(1), c));
            indexJ = (indexJ+1)%vertexData.size();
        }
    }    

    std::vector<cv::Point2f> dualPoints(constraints.size(), cv::Point2f(0,0));
    for (size_t i=0; i<constraints.size(); i++) {
        dualPoints[i].x = constraints[i](0)/constraints[i](2);
        dualPoints[i].y = constraints[i](1)/constraints[i](2);
    }
    
    std::vector<cv::Point2f> dualVertex, finalVertex;
    cv::convexHull(dualPoints,dualVertex,true,false);

    for (size_t i=0; i<dualVertex.size(); i++) {
        int iplus1 = (i+1)%dualVertex.size();
        cv::Point2f rayi = dualVertex[iplus1] - dualVertex[i];
        float c = rayi.y*dualVertex[i].x - rayi.x*dualVertex[i].y;
        finalVertex.push_back(cv::Point2f(interior_x+rayi.y/c, interior_y-rayi.x/c));
    }

    unsigned int size = finalVertex.size();
    hPoly.resize(4, size);
    for (unsigned int i = 0; i < size; i++){
    int iplus1 = (i+1)%size;
    cv::Point2f rayi = finalVertex[iplus1] - finalVertex[i];           
    hPoly.col(i).tail<2>()  = Eigen::Vector2d(finalVertex[i].x, finalVertex[i].y);  // the points on the plane
    hPoly.col(i).head<2>()  = Eigen::Vector2d(-rayi.y, rayi.x); // outside
    }
      
  }
  Eigen::MatrixXd getDilateRec(Eigen::Vector3d state, map_util::OccMapUtil gridmap, int maxIndexL = 5, int maxIndexW = 8){

    double resolution = gridmap.getRes();
    double step = resolution * 1.0;
    double limitBoundHalfL = maxIndexL * resolution;
    double limitBoundHalfW = maxIndexW * resolution;
    //generate a rectangle for this state px py yaw
    //generate a hPoly
    Eigen::Matrix<int,4,1> NotFinishTable = Eigen::Matrix<int,4,1>(1,1,1,1);      
    Eigen::Vector2d sourcePt = state.head(2);
    double yaw = state[2];
    Eigen::Vector4d expandLength;
    expandLength << 0.0, 0.0, 0.0, 0.0;
    Eigen::Matrix2d egoR;
      egoR << cos(yaw), -sin(yaw),
              sin(yaw), cos(yaw);
    //dcr width length
    while(NotFinishTable.norm()>0){ 
    //+dx -dy -dx  +dy  
        for(int i = 0; i<4; i++){
            if(!NotFinishTable[i]) continue;
            //get the new source and vp
            Eigen::Vector4d  tmp_expandLength = expandLength;
            Eigen::Vector2d cor1, cor2, cor3, cor4, cor5;
            bool isocc = false;
            switch (i)
            {
            //+dx
            case 0:
                tmp_expandLength[0] += step;
                cor1 = sourcePt + egoR * Eigen::Vector2d(tmp_expandLength[0],tmp_expandLength[3]);
                cor2 = sourcePt + egoR * Eigen::Vector2d(tmp_expandLength[0],-tmp_expandLength[1]);
                cor3 = sourcePt + egoR * Eigen::Vector2d(-tmp_expandLength[2],-tmp_expandLength[1]);
                cor4 = sourcePt + egoR * Eigen::Vector2d(-tmp_expandLength[2],tmp_expandLength[3]);
                cor5 = cor1;
                //1 new1 new1 new2 new2 2
                isocc = gridmap.isBlocked(cor1, cor2);
                if(isocc){
                    NotFinishTable[i] = 0.0;
                    break;
                }
                isocc = gridmap.isBlocked(cor2, cor3);
                if(isocc){
                    NotFinishTable[i] = 0.0;
                    break;
                }
                isocc = gridmap.isBlocked(cor3, cor4);
                if(isocc){
                    NotFinishTable[i] = 0.0;
                    break;
                }
                isocc = gridmap.isBlocked(cor4, cor5);
                if(isocc){
                    NotFinishTable[i] = 0.0;
                    break;
                }
                if(tmp_expandLength[0] > limitBoundHalfL){
                    NotFinishTable[i] = 0.0;
                    break;
                }
                expandLength = tmp_expandLength;
                break;
            //-dy
            case 1:
                tmp_expandLength[1] += step;
                cor1 = sourcePt + egoR * Eigen::Vector2d(tmp_expandLength[0],tmp_expandLength[3]);
                cor2 = sourcePt + egoR * Eigen::Vector2d(tmp_expandLength[0],-tmp_expandLength[1]);
                cor3 = sourcePt + egoR * Eigen::Vector2d(-tmp_expandLength[2],-tmp_expandLength[1]);
                cor4 = sourcePt + egoR * Eigen::Vector2d(-tmp_expandLength[2],tmp_expandLength[3]);
                cor5 = cor1;
                //1 new1 new1 new2 new2 2
                isocc = gridmap.isBlocked(cor1, cor2);
                if(isocc){
                    NotFinishTable[i] = 0.0;
                    break;
                }
                isocc = gridmap.isBlocked(cor2, cor3);
                if(isocc){
                    NotFinishTable[i] = 0.0;
                    break;
                }
                isocc = gridmap.isBlocked(cor3, cor4);
                if(isocc){
                    NotFinishTable[i] = 0.0;
                    break;
                }
                isocc = gridmap.isBlocked(cor4, cor5);
                if(isocc){
                    NotFinishTable[i] = 0.0;
                    break;
                }
                if(tmp_expandLength[1] > limitBoundHalfW){
                    NotFinishTable[i] = 0.0;
                    break;
                }
                expandLength = tmp_expandLength;
                break;
            //-dx
            case 2:
                tmp_expandLength[2] += step;
                cor1 = sourcePt + egoR * Eigen::Vector2d(tmp_expandLength[0],tmp_expandLength[3]);
                cor2 = sourcePt + egoR * Eigen::Vector2d(tmp_expandLength[0],-tmp_expandLength[1]);
                cor3 = sourcePt + egoR * Eigen::Vector2d(-tmp_expandLength[2],-tmp_expandLength[1]);
                cor4 = sourcePt + egoR * Eigen::Vector2d(-tmp_expandLength[2],tmp_expandLength[3]);
                cor5 = cor1;
                //1 new1 new1 new2 new2 2
                isocc = gridmap.isBlocked(cor1, cor2);
                if(isocc){
                    NotFinishTable[i] = 0.0;
                    break;
                }
                isocc = gridmap.isBlocked(cor2, cor3);
                if(isocc){
                    NotFinishTable[i] = 0.0;
                    break;
                }
                isocc = gridmap.isBlocked(cor3, cor4);
                if(isocc){
                    NotFinishTable[i] = 0.0;
                    break;
                }
                isocc = gridmap.isBlocked(cor4, cor5);
                if(isocc){
                    NotFinishTable[i] = 0.0;
                    break;
                }
                if(tmp_expandLength[2] > limitBoundHalfL){
                    NotFinishTable[i] = 0.0;
                    break;
                }
                expandLength = tmp_expandLength;
                break;
            //dy
            case 3:
                tmp_expandLength[3] += step;
                cor1 = sourcePt + egoR * Eigen::Vector2d(tmp_expandLength[0],tmp_expandLength[3]);
                cor2 = sourcePt + egoR * Eigen::Vector2d(tmp_expandLength[0],-tmp_expandLength[1]);
                cor3 = sourcePt + egoR * Eigen::Vector2d(-tmp_expandLength[2],-tmp_expandLength[1]);
                cor4 = sourcePt + egoR * Eigen::Vector2d(-tmp_expandLength[2],tmp_expandLength[3]);
                cor5 = cor1;
                //1 new1 new1 new2 new2 2
                isocc = gridmap.isBlocked(cor1, cor2);
                if(isocc){
                    NotFinishTable[i] = 0.0;
                    break;
                }
                isocc = gridmap.isBlocked(cor2, cor3);
                if(isocc){
                    NotFinishTable[i] = 0.0;
                    break;
                }
                isocc = gridmap.isBlocked(cor3, cor4);
                if(isocc){
                    NotFinishTable[i] = 0.0;
                    break;
                }
                isocc = gridmap.isBlocked(cor4, cor5);
                if(isocc){
                    NotFinishTable[i] = 0.0;
                    break;
                }
                if(tmp_expandLength[3] > limitBoundHalfW){
                    NotFinishTable[i] = 0.0;
                    break;
                }
                expandLength = tmp_expandLength;
                break;
            }   
        }
    }
    Eigen::MatrixXd poly;
    poly.resize(4,4);
    std::vector<Eigen::Vector2d> cors;
    cors.resize(4);
    cors[0] = sourcePt + egoR * Eigen::Vector2d(expandLength[0],expandLength[3]);
    cors[1] = sourcePt + egoR * Eigen::Vector2d(expandLength[0],-expandLength[1]);
    cors[2] = sourcePt + egoR * Eigen::Vector2d(-expandLength[2],-expandLength[1]);
    cors[3] = sourcePt + egoR * Eigen::Vector2d(-expandLength[2],expandLength[3]);
    poly.col(0) << cors[0], cos(yaw), sin(yaw);
    poly.col(1) << cors[1], sin(yaw), -cos(yaw);
    poly.col(2) << cors[2], -cos(yaw), -sin(yaw);
    poly.col(3) << cors[3],-sin(yaw), cos(yaw);

    return poly;

}
bool ifInside(Eigen::Vector2d pt, Eigen::MatrixXd poly){
    for(int i = 0; i < poly.cols(); i++){
        Eigen::Vector2d p = poly.col(i).head(2);
        Eigen::Vector2d n = poly.col(i).tail(2);
        if((pt-p).dot(n)>0){
            return false;
        }
    }
    return true;

}
} // namespace plan_utils