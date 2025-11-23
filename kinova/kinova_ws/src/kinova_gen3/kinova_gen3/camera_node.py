from kinova_gen3_interfaces.srv import GetCoordinates
import rclpy
from rclpy.node import Node
import cv2
import numpy as np

class Coordinates(Node):
    def __init__ (self):
        super().__init__('coordinate_provider')
        self.srv = self.create_service(
            GetCoordinates,
            'get_coordinates',
            self.get_coordinates_callback
        )
        self.coordinates_ready = False
        self.latest_coords = []
        
        self.get_logger().info('Service ready at: /get_coordinates')
        self.get_logger().info('Waiting for coordinate requests...')
        self.get_logger().info('='*60)

def detect_blocks_from_image(self, image_path=""):
        """
        Detect blocks in image and return coordinates
        
        TODO: Implement OpenCV detection here
        - Load image
        - Run inference/detection
        - Convert pixel coords to world coords
        - Return coordinate list
        """
        self.get_logger().info(f'Processing image: {image_path if image_path else "live camera"}')
        
        # FAKE COORDINATES FOR NOW
        # Replace this with your actual OpenCV detection code
        coords = [
            [0.075, 0.265],  # Blue reference square
            [0.45, 0.15],    # Block 1
            [0.50, 0.20],    # Block 2
            [0.40, 0.25],    # Block 3
        ]
        
        self.get_logger().info(f'Detected {len(coords)} blocks')
        for i, coord in enumerate(coords):
            self.get_logger().info(f'  Block {i}: x={coord[0]:.3f}m, y={coord[1]:.3f}m')
        
        return coords

def get_coordinates_callback(self, request, response):
        """
       Service callback that processes coordinate requests
        
        This is where you'll add your OpenCV vision code
        """
        self.get_logger().info('Received coordinate request')
        
        try:
            # Get coordinates from vision system
            coords = self.detect_blocks_from_image(request.image_path)
            
            # Format response
            response.x_coords = [coord[0] for coord in coords]
            response.y_coords = [coord[1] for coord in coords]
            response.num_blocks = len(coords)
            response.success = True
            response.message = f'Successfully detected {len(coords)} blocks'
            
            self.get_logger().info(f'Returning {len(coords)} coordinates')
            
        except Exception as e:
            self.get_logger().error(f'Error detecting blocks: {str(e)}')
            response.success = False
            response.message = f'Error: {str(e)}'
            response.num_blocks = 0
        
        return response

def main(args=None):
    rclpy.init(args=args)
    node = Coordinates()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down vision node...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()