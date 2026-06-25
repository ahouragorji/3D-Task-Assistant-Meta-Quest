using UnityEngine;

namespace Alpha.Parsers
{
    public class IndicatorArrowParser : IToolParser
    {
        public string ToolName => "indicator_arrow";

        private GameObject _up, _down, _front, _back, _left, _right;
        private float _offset;

        // The constructor takes the prefabs passed from the main Receiver
        public IndicatorArrowParser(GameObject up, GameObject down, GameObject front, GameObject back, GameObject left, GameObject right, float offset)
        {
            _up = up; _down = down; _front = front; _back = back; 
            _left = left; _right = right; _offset = offset;
        }

       public ParsedSpawnData Parse(QuestInstructionReceiver.AROverlay overlay, Vector3 userSavedPos)
        {
            // 1. Extract the placement rule from the tool settings array
            string placement = GetParam(overlay.tool_settings, "placement_rule", "up");

            // 2. Determine which specific arrow prefab to spawn
            GameObject prefab = _up;
            switch (placement)
            {
                case "down": prefab = _down; break;
                case "front": prefab = _front; break;
                case "back": prefab = _back; break;
                case "left": prefab = _left; break;
                case "right": prefab = _right; break;
            }

            // 3. Calculate Spatial Direction Vectors
            Vector3 center = new Vector3(overlay.worldX, overlay.worldY, overlay.worldZ);
            
            // Vector pointing from the object to the user (flattened on the Y axis)
            Vector3 objectToUser = userSavedPos - center;
            objectToUser.y = 0;
            objectToUser.Normalize();

            Vector3 userToObject = -objectToUser;
            
            // Cross product gives us true Left/Right relative to the user looking at the object
            Vector3 userRight = Vector3.Cross(Vector3.up, userToObject).normalized;

            // Map the placement string to the mathematical vector direction
            Vector3 dir = Vector3.up;
            switch (placement)
            {
                case "down": dir = Vector3.down; break;
                case "front": dir = objectToUser; break;
                case "back": dir = userToObject; break;
                case "left": dir = -userRight; break;
                case "right": dir = userRight; break;
            }

            // 4. Find the edge of the bounding box in that specific direction
            float edgeDistance = 0f;
            if (overlay.bboxCorners != null && overlay.bboxCorners.Length == 8)
            {
                foreach (var corner in overlay.bboxCorners)
                {
                    Vector3 centerToCorner = corner.ToVector3() - center;
                    // Project the corner onto our chosen direction using a Dot Product
                    float projection = Vector3.Dot(centerToCorner, dir);
                    
                    // Save the extent that sticks out the furthest
                    if (projection > edgeDistance) edgeDistance = projection;
                }
            }

            // Calculate final position: Center + Edge of Box + Configurable Padding
            Vector3 finalPos = center + (dir * (edgeDistance + _offset));

            // 5. Calculate Rotation (Point the arrow at the center of the object)
            Quaternion rot = (finalPos != center)
                ? Quaternion.LookRotation(center - finalPos, Vector3.up)
                : Quaternion.identity;

            return new ParsedSpawnData
            {
                PrefabToSpawn = prefab,
                Position = finalPos,
                Rotation = rot
            };
        }
        
        private string GetParam(QuestInstructionReceiver.FeatureParameter[] parameters, string key, string fallback)
        {
            if (parameters == null) return fallback;
            foreach (var p in parameters) if (p.key == key) return p.value;
            return fallback;
        }

    }
}