# Compiler vars
CUDA := nvcc
 CUDAFLAGS := -arch=sm_60 -O -Wno-deprecated-gpu-targets
#
# # Paths
BUILD_DIR := sobel_cuda
SRC_DIR := sobel_cuda 
SRC_PART1_DIR := $(SRC_DIR)/Part1
SRC_PART2_DIR := $(SRC_DIR)
#
# Targets
#TARGET_PART1 := lab4p1
 TARGET_PART2 := lab4p2
#
 TARGET_EXECUTABLES := \
 		$(TARGET_PART2) \
#
# 		# File lists for Part 2
 		SRCS_PART2 := $(shell find $(SRC_PART2_DIR) -name *.c)
 		OBJ_PART2 := $(SRCS_PART2:%=$(BUILD_DIR)/%.o)
 		LIB_OBJ_PART2 := $(SRC_PART2_DIR)/bmpReader.o
#
#
 		all: $(TARGET_EXECUTABLES)
#
# 		$(TARGET_PART1):
# 			$(CUDA) $(CUDAFLAGS) -o $@ $(SRC_PART1_DIR)/maxwell_griffin_$@.cu
#
 			$(TARGET_PART2): $(OBJ_PART2)
				@mkdir -p $(SRC_PART2_DIR)
				$(CUDA) $(CUDAFLAGS) -dc $(SRC_PART2_DIR)/sobel_edge -o $(SRC_PART2_DIR)/sobel_edge_$@.o
				$(CUDA) $(CUDAFLAGS) -o $@ $(OBJ_PART2) $(LIB_OBJ_PART2) $(SRC_PART2_DIR)/maxwell_griffin_$@.o
#
# 						# c source
						$(BUILD_DIR)/%.c.o: %.c
							mkdir -p $(dir $@)
							$(CUDA) $(CUDAFLAGS) -x c -c $< -o $@
#
