/*
 * Student Name: Mehmet Yasin ŞEREMET
 * Student Number: 2017400102
 * Compile Status: Compiling
 * Program Status: Working
 * Notes: --- (Works normally with specified commands in the description.)
 */

#include <iostream>
#include <mpi.h>
#include <stdlib.h>
#include <sstream>
#include <fstream>
#include <unistd.h>
#include <bits/stdc++.h>

using namespace std;

int main(int argc, char *argv[])
{
    int rank; //process rank
    int size; //process size
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int totalRank;	    //total number of processors (1 master, totalRank-1 slave)
	int instanceNumb;   //total number of instances
	int featureNumb;    //total number of features
	int iterationNumb;      //total number of iterationNumbs
	int topFeaturesNumb;    //total number of top features

	string inst, token1, token2, token3, token4; //strings to use in data reading

	//reading input file and initializing the required numbers
	ifstream inst_file (argv[1]);
	getline(inst_file,inst);
	totalRank = stoi(inst);
	getline(inst_file,inst);
	stringstream splitline(inst);
	splitline >> token1 >> token2 >> token3 >> token4;
	instanceNumb=stoi(token1);
	featureNumb=stoi(token2);
	iterationNumb=stoi(token3);
	topFeaturesNumb=stoi(token4);

	//arrSent is the array distributed by master to each slave. It is 3-dimensional. First dimension indicates rank
	//of slave which the related data will be sent to. Second dimension indicates instance number or line will be sent
	//to related slave. Third dimension indicates the feature number or class label.
	double arrSent[totalRank][instanceNumb/(totalRank-1)][featureNumb+1];

	//arrReceived is the array received by each slave. It is 2-dimensional. First dimension indicates instance number
	//or line will be received by related slave. Second dimension indicates feature number or class label.
	double arrReceived[instanceNumb/(totalRank-1)][featureNumb+1];
	
	double weight[featureNumb]={0};     //weight array for features
	int slaveTopFeatures[topFeaturesNumb];    //array for top features with highest weights(sent by each slave to master)
	int masterTopFeatures[topFeaturesNumb*totalRank];    //array for result of top features(gathered by master)

	//master process
	if(rank==0){
		string inst, token1; //strings to use in data reading

		//reading input file and put all the data in arrSent number by number since it will send the required data to related slave
		if (inst_file.is_open()) {
			ifstream inst_file (argv[1]);
			getline(inst_file,inst);
			getline(inst_file,inst);
			
			for(int i=1; i<totalRank; i++){
				for(int j=0; j<(instanceNumb/(totalRank-1)); j++){
					getline(inst_file, inst);
					stringstream splitline(inst);
					
					for(int k=0; k<featureNumb+1; k++){
						splitline>>token1;
						arrSent[i][j][k]=stod(token1);
					}		
				}
			}
		}
	}
	
	
	//sending data from master array arrSent to arrReceived array on each slave
    MPI_Scatter(arrSent,(instanceNumb/(totalRank-1))*(featureNumb+1),MPI_DOUBLE,arrReceived,(instanceNumb/(totalRank-1))*(featureNumb+1),MPI_DOUBLE,0,MPI_COMM_WORLD);
	
	int masterSignal = 1; //variable to end slaves after master is finished
    while(masterSignal){

        //slave process
        if(rank!= 0){

            //initializing the slaveTopFeatures to -1
            for(int i=0;i<topFeaturesNumb;i++){
                slaveTopFeatures[i]=-1;
            }

            //relief algorithm starts
			for(int k=0; k<iterationNumb; k++){
				int whichTarget=k; //index of target instance
				double target[featureNumb+1]; //feature array of target instance

				//initializing target instance
				for(int i=0; i<featureNumb+1; i++){
					target[i]=arrReceived[k][i];
				}
							
				double distArrHit[instanceNumb/(totalRank-1)]; //array of distance between target and hit instances
				double distArrMiss[instanceNumb/(totalRank-1)]; //array of distance between target and miss instances

				//calculating the distance between target and hit-miss instances
				//putting the results to related arrays
				for(int i=0;i<instanceNumb/(totalRank-1); i++){
					double tempHit=0,tempMiss=0;
					double totalHit=0, totalMiss=0;
					distArrHit[i]=-1;
					distArrMiss[i]=-1;
					
					if(i==whichTarget){
						
					}
					else{
						if(arrReceived[i][featureNumb]==target[featureNumb]){ //for hits
							for(int j=0; j<featureNumb; j++){
								tempHit = target[j] - arrReceived[i][j];
								if(tempHit<0){
									tempHit=-tempHit;
								}
								totalHit=totalHit+tempHit;
							}
							distArrHit[i]=totalHit;
						}
						else{   //for misses
							for(int j=0; j<featureNumb; j++){
								tempMiss = target[j] - arrReceived[i][j];
								if(tempMiss<0){
									tempMiss=-tempMiss;
								}
								totalMiss=totalMiss+tempMiss;
							}
							distArrMiss[i]=totalMiss;
						}
					}
				}

				double smallestHit=999999, smallestMiss=999999; //smallest distances for hits and misses
				int resultHit=-1, resultMiss=-1;    //indexes of smallest distance instances

				//finding the indexes(line numbers) of instances with smallest distance for hit and miss
				for(int i=0;i<instanceNumb/(totalRank-1);i++){
					if(distArrHit[i]<smallestHit && distArrHit[i]!=-1){ //for hit
						smallestHit=distArrHit[i];
						resultHit=i;
					}
					if(distArrMiss[i]<smallestMiss && distArrMiss[i]!=-1){  //for miss
						smallestMiss=distArrMiss[i];
						resultMiss=i;
					}
				}


                double diffHit=0;   //equal to diff(A,Ri,H)/m in the weight formula
                double diffMiss=0;  //equal to diff(A,Ri,M)/m in the weight formula
                double maxA=-999999;//equal to max(A) in the diff formula
                double minA=999999; //equal to min(A) in the diff formula

                //calculating weights and filling the weight array for all features
                for(int i=0;i<featureNumb;i++){
                    for(int j=0;j<instanceNumb/(totalRank-1);j++){
                        if(arrReceived[j][i]>maxA){
                            maxA=arrReceived[j][i];
                        }
                        if(arrReceived[j][i]<minA){
                            minA=arrReceived[j][i];
                        }
                    }

                    diffHit = ((target[i]-arrReceived[resultHit][i])/(maxA-minA))/iterationNumb;
                    diffMiss = ((target[i]-arrReceived[resultMiss][i])/(maxA-minA))/iterationNumb;

                    if(diffHit<0){
                        diffHit=-diffHit;
                    }
                    if(diffMiss<0){
                        diffMiss=-diffMiss;
                    }

                    weight[i]=weight[i]-diffHit+diffMiss; //the main formula to calculate weight

                    maxA=-999999; minA=999999;
                }
			}
			
			
			double tempMax=-999999; //max element in the weight array
			int maxIndex=-1;    //index of max element in the weight array

			//putting the top features of the slave to the slaveTopFeatures array
			for(int j=0;j<topFeaturesNumb;j++){
				for(int i=0;i<featureNumb;i++){
					if(weight[i]>tempMax){
						tempMax=weight[i];
						maxIndex=i;
					}
				}
				slaveTopFeatures[j]=maxIndex;
				weight[maxIndex]=-999999;
				tempMax=-999999;
				maxIndex=-1;
			}

			//sorting the top features for a nice output format
			int n = sizeof(slaveTopFeatures) / sizeof(slaveTopFeatures[0]);
			sort(slaveTopFeatures, slaveTopFeatures + n);

			//printing the slave's top features
			cout<<"Slave P"<<rank<<": ";
			for(int i=0;i<topFeaturesNumb;i++){
				cout<<slaveTopFeatures[i]<<" ";
			}
			cout<<endl;

			//avoiding the master's output disorder
			usleep(5000);
        }

        //sending data from slaveTopFeatures array on each slave to master array masterTopFeatures
        MPI_Gather(slaveTopFeatures,topFeaturesNumb,MPI_INT,masterTopFeatures,topFeaturesNumb,MPI_INT,0,MPI_COMM_WORLD);

        //master process
        if(rank==0){
			set<int> resultSet; //set for master's results

			//putting the results to set to remove the duplicates
			for(int i=0;i<topFeaturesNumb*totalRank;i++){
				resultSet.insert(masterTopFeatures[i]);
			}

			//printing the master's top features
			set<int>::iterator it;
			cout<<"Master P0: ";
			for(it=resultSet.begin(); it!=resultSet.end(); ++it){
				if(*it!=-1){
				cout<<*it<<" ";
				}
			}
			cout<<endl;
			
            masterSignal=0;
        }

        //broadcasting to end processes
        MPI_Bcast(&masterSignal, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }	

	MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return(0);
}


