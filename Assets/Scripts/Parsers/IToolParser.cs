using UnityEngine;

namespace Alpha.Parsers
{
    // The data container passed back to the spawner
    public class ParsedSpawnData
    {
        public GameObject PrefabToSpawn;
        public Vector3 Position;
        public Quaternion Rotation;
    }

    // The contract every parser must follow
    public interface IToolParser
    {
        string ToolName { get; }
        ParsedSpawnData Parse(QuestInstructionReceiver.AROverlay overlay, Vector3 userSavedPos);
    }
}