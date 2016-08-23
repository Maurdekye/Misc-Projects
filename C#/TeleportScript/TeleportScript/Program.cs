using GTA;
using GTA.Math;
using GTA.Native;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace TeleportScript
{
    class MainScript : Script
    {
        public MainScript()
        {
            KeyUp += keyval;
        } 

        void keyval(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.F8)
            {
                Ped PlayerPed = Game.Player.Character;
                Function.Call(Hash.GET_ENTITY_COORDS, );
            }
        }
    }
}
