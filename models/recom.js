const mongoose= require ('mongoose');
const Schema= mongoose.Schema;

const recomSchema= new mongoose.Schema({
    userId:{
        type:String,
        required:true
    },
    itemId:{
        type:String,
        required:true,
    },
    rating:{ 
        type:Number,
        required:true
    }

    
})

const Recom= mongoose.model('Recom',recomSchema);

module.exports={Recom};