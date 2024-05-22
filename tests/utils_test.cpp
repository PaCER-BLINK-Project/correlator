#include <iostream>
#include <cstring>
#include "common.hpp"
#include "../src/utils.hpp"


std::string dataRootDir {std::getenv(ENV_DATA_ROOT_DIR)};



void test_read_data_from_file(){
    char *buffer {nullptr};
    size_t size {0};
    read_data_from_file(dataRootDir + "/simple/text_input.txt", buffer, size);
    if(size != 17) throw TestFailed("'test_read_data_from_file' (1) failed.");
    buffer[17] = '\0';
    if(strcmp(buffer, "simple text input")) throw TestFailed("'test_read_data_from_file' (2) failed.");
    std::cout << "'test_read_data_from_file' passed." << std::endl;
}



void test_parse_timespec(){
    if(
        parse_timespec("1s") != 1.0 ||
        parse_timespec("2ds") != 0.2 ||
        parse_timespec("10ms") != 0.01 ||
        parse_timespec("4cs") != 0.04 ||
        parse_timespec("0.4s") != 0.4
    ) throw TestFailed("'test_parse_timespec' (1) failed.");
    double tsp = -4.0;
    try{
        tsp = parse_timespec("4kg");
    }catch (std::invalid_argument ex){
        tsp = -5.0;
    }
    if(tsp != -5.0) throw TestFailed("'test_parse_timespec' (2) failed.");
     try{
        tsp = parse_timespec("0.3.4s");
    }catch (std::invalid_argument ex){
        tsp = -6.0;
    }
    if(tsp != -6.0) throw TestFailed("'test_parse_timespec' (3) failed.");
    std::cout << "'test_parse_timespec' passed." << std::endl;
}           
            


int main(void){

    try{
        
        test_parse_timespec();
        test_read_data_from_file();

    } catch (TestFailed ex){
        std::cerr << ex.what() << std::endl;
        return 1;
    }
    
    std::cout << "All tests passed." << std::endl;
    return 0;
}
